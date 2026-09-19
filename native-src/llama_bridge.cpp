#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

#include "llama.h"

#define LOG_TAG "llama_bridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

struct ModelContext {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    llama_sampler *sampler = nullptr;
};

static bool backendInitialized = false;

extern "C" JNIEXPORT jlong JNICALL
Java_com_yadaventerprise_phoneserver_LlmEngine_nativeLoadModel(
        JNIEnv *env, jobject, jstring modelPathJ) {

    if (!backendInitialized) {
        llama_backend_init();
        backendInitialized = true;
    }

    const char *modelPath = env->GetStringUTFChars(modelPathJ, nullptr);
    llama_model_params modelParams = llama_model_default_params();
    llama_model *model = llama_load_model_from_file(modelPath, modelParams);
    env->ReleaseStringUTFChars(modelPathJ, modelPath);

    if (model == nullptr) {
        LOGE("Failed to load model");
        return 0;
    }

    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = 2048;
    ctxParams.n_batch = 512;
    ctxParams.n_threads = 4;
    ctxParams.n_threads_batch = 4;

    llama_context *ctx = llama_new_context_with_model(model, ctxParams);
    if (ctx == nullptr) {
        LOGE("Failed to create context");
        llama_free_model(model);
        return 0;
    }

    llama_sampler_chain_params samplerParams = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(samplerParams);
    // llama.cpp added a fifth penalty_decay argument in newer revisions.
    llama_sampler_chain_add(
            sampler, llama_sampler_init_penalties(64, 1.1f, 0.0f, 0.0f, 1.0f));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    auto *modelContext = new ModelContext();
    modelContext->model = model;
    modelContext->ctx = ctx;
    modelContext->sampler = sampler;

    LOGI("Model loaded successfully");
    return reinterpret_cast<jlong>(modelContext);
}

extern "C" JNIEXPORT void JNICALL
Java_com_yadaventerprise_phoneserver_LlmEngine_nativeUnloadModel(
        JNIEnv *env, jobject, jlong handle) {

    auto *modelContext = reinterpret_cast<ModelContext *>(handle);
    if (modelContext == nullptr) return;

    if (modelContext->sampler) llama_sampler_free(modelContext->sampler);
    if (modelContext->ctx) llama_free(modelContext->ctx);
    if (modelContext->model) llama_free_model(modelContext->model);
    delete modelContext;
}

extern "C" JNIEXPORT void JNICALL
Java_com_yadaventerprise_phoneserver_LlmEngine_nativeGenerate(
        JNIEnv *env, jobject, jlong handle, jstring promptJ, jobject tokenSink) {

    auto *modelContext = reinterpret_cast<ModelContext *>(handle);
    if (modelContext == nullptr) {
        LOGE("nativeGenerate called with null model context");
        return;
    }

    jclass sinkClass = env->GetObjectClass(tokenSink);
    jmethodID onTokenMethod = env->GetMethodID(sinkClass, "onToken", "(Ljava/lang/String;)V");
    if (onTokenMethod == nullptr) {
        LOGE("Could not find TokenSink.onToken(String)");
        return;
    }

    const char *promptChars = env->GetStringUTFChars(promptJ, nullptr);
    std::string prompt(promptChars);
    env->ReleaseStringUTFChars(promptJ, promptChars);

    std::string formattedPrompt =
            "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";

    const llama_model *model = modelContext->model;
    llama_context *ctx = modelContext->ctx;
    llama_sampler *sampler = modelContext->sampler;
    const llama_vocab *vocab = llama_model_get_vocab(model);

    const int maxTokens = 1024;
    std::vector<llama_token> tokens(maxTokens);
    int nTokens = llama_tokenize(
            vocab, formattedPrompt.c_str(), (int32_t) formattedPrompt.size(),
            tokens.data(), maxTokens, true, true);

    if (nTokens < 0) {
        LOGE("Prompt too long for context");
        return;
    }
    tokens.resize(nTokens);

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        LOGE("llama_decode failed on prompt");
        return;
    }

    const int maxNewTokens = 256;
    for (int i = 0; i < maxNewTokens; i++) {
        llama_token newToken = llama_sampler_sample(sampler, ctx, -1);

        if (llama_vocab_is_eog(llama_model_get_vocab(model), newToken)) {
            break;
        }

        char pieceBuffer[256];
        int pieceLen = llama_token_to_piece(
                llama_model_get_vocab(model), newToken, pieceBuffer, sizeof(pieceBuffer),
                0, false);

        if (pieceLen > 0) {
            std::string piece(pieceBuffer, pieceLen);
            jstring pieceJ = env->NewStringUTF(piece.c_str());
            env->CallVoidMethod(tokenSink, onTokenMethod, pieceJ);
            env->DeleteLocalRef(pieceJ);
        }

        llama_sampler_accept(sampler, newToken);

        llama_batch nextBatch = llama_batch_get_one(&newToken, 1);
        if (llama_decode(ctx, nextBatch) != 0) {
            LOGE("llama_decode failed mid-generation");
            break;
        }
    }
}
