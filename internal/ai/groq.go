package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const (
	groqBaseURL             = "https://api.groq.com/openai/v1"
	groqTranscriptionURL    = groqBaseURL + "/audio/transcriptions"
	groqChatCompletionsURL  = groqBaseURL + "/chat/completions"
)

// GroqProvider implements AIProvider using Groq API
type GroqProvider struct {
	httpClient   *http.Client
	apiKey       string
	conf         ProviderConfig
	whisperModel string
}

// Groq API request/response types

type groqChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type groqChatRequest struct {
	Model    string            `json:"model"`
	Messages []groqChatMessage `json:"messages"`
}

type groqChatResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Error *groqErrorResponse `json:"error,omitempty"`
}

type groqTranscriptionResponse struct {
	Text  string             `json:"text"`
	Error *groqErrorResponse `json:"error,omitempty"`
}

type groqErrorResponse struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// NewGroqProvider creates a new Groq API provider
func NewGroqProvider(httpClient *http.Client, apiKey string, conf ProviderConfig, whisperModel string) *GroqProvider {
	return &GroqProvider{
		httpClient:   httpClient,
		apiKey:       apiKey,
		conf:         conf,
		whisperModel: whisperModel,
	}
}

// AudioToText transcribes audio file to text using Groq Whisper API
func (g *GroqProvider) AudioToText(ctx context.Context, audioPath string) (string, error) {
	var lastErr error

	for attempt := 1; attempt <= g.conf.PrimaryModelRetries; attempt++ {
		text, err := g.transcribeAudio(ctx, audioPath)
		if err == nil && text != "" {
			return text, nil
		}
		if err != nil {
			lastErr = err
		} else {
			lastErr = fmt.Errorf("Whisper API вернул пустой ответ")
		}

		if isRetryableError(lastErr) && attempt < g.conf.PrimaryModelRetries {
			time.Sleep(g.conf.RetryDelay)
			continue
		}
		break
	}

	return "", fmt.Errorf("ошибка транскрипции аудио: %w", lastErr)
}

func (g *GroqProvider) transcribeAudio(ctx context.Context, audioPath string) (string, error) {
	file, err := os.Open(audioPath)
	if err != nil {
		return "", fmt.Errorf("не удалось открыть аудиофайл: %w", err)
	}
	defer file.Close()

	// Create multipart form
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add file field
	filename := filepath.Base(audioPath)
	part, err := writer.CreateFormFile("file", filename)
	if err != nil {
		return "", fmt.Errorf("не удалось создать form file: %w", err)
	}
	if _, err := io.Copy(part, file); err != nil {
		return "", fmt.Errorf("не удалось скопировать файл в форму: %w", err)
	}

	// Add model field
	if err := writer.WriteField("model", g.whisperModel); err != nil {
		return "", fmt.Errorf("не удалось добавить поле model: %w", err)
	}

	// Add response_format field
	if err := writer.WriteField("response_format", "json"); err != nil {
		return "", fmt.Errorf("не удалось добавить поле response_format: %w", err)
	}

	if err := writer.Close(); err != nil {
		return "", fmt.Errorf("не удалось закрыть multipart writer: %w", err)
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, groqTranscriptionURL, &buf)
	if err != nil {
		return "", fmt.Errorf("не удалось создать запрос: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+g.apiKey)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Execute request
	resp, err := g.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("ошибка выполнения запроса к Whisper API: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("ошибка чтения ответа: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Whisper API вернул статус %d: %s", resp.StatusCode, string(body))
	}

	var transcription groqTranscriptionResponse
	if err := json.Unmarshal(body, &transcription); err != nil {
		return "", fmt.Errorf("ошибка декодирования ответа: %w", err)
	}

	if transcription.Error != nil {
		return "", fmt.Errorf("ошибка Whisper API: %s", transcription.Error.Message)
	}

	return transcription.Text, nil
}

// SummarizeText creates a summary using Groq Chat Completions API
func (g *GroqProvider) SummarizeText(ctx context.Context, textToSummarize, promptTemplate string) (string, error) {
	var lastErr error
	userPrompt := fmt.Sprintf(promptTemplate, textToSummarize)

	messages := []groqChatMessage{
		{Role: "system", Content: g.conf.SystemPrompt},
		{Role: "user", Content: userPrompt},
	}

	// Try primary model
	for attempt := 1; attempt <= g.conf.PrimaryModelRetries; attempt++ {
		text, err := g.chatCompletion(ctx, g.conf.PrimaryModel, messages)
		if err == nil && text != "" {
			return text, nil
		}
		if err != nil {
			lastErr = err
		} else {
			lastErr = fmt.Errorf("Chat API вернул пустой ответ")
		}

		if isRetryableError(lastErr) && attempt < g.conf.PrimaryModelRetries {
			time.Sleep(g.conf.RetryDelay)
			continue
		}
		break
	}

	// Try fallback model
	for attempt := 1; attempt <= g.conf.FallbackModelRetries; attempt++ {
		text, err := g.chatCompletion(ctx, g.conf.FallbackModel, messages)
		if err == nil && text != "" {
			return text, nil
		}
		if err != nil {
			lastErr = err
		} else {
			lastErr = fmt.Errorf("Chat API вернул пустой ответ")
		}

		if isRetryableError(lastErr) && attempt < g.conf.FallbackModelRetries {
			time.Sleep(g.conf.RetryDelay)
			continue
		}
		break
	}

	return "", fmt.Errorf("все попытки суммирования не удались, последняя ошибка: %w", lastErr)
}

func (g *GroqProvider) chatCompletion(ctx context.Context, model string, messages []groqChatMessage) (string, error) {
	reqBody := groqChatRequest{
		Model:    model,
		Messages: messages,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("ошибка сериализации запроса: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, groqChatCompletionsURL, bytes.NewBuffer(jsonBody))
	if err != nil {
		return "", fmt.Errorf("не удалось создать запрос: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+g.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := g.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("ошибка выполнения запроса к Chat API: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("ошибка чтения ответа: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Chat API вернул статус %d: %s", resp.StatusCode, string(body))
	}

	var chatResp groqChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return "", fmt.Errorf("ошибка декодирования ответа: %w", err)
	}

	if chatResp.Error != nil {
		return "", fmt.Errorf("ошибка Chat API: %s", chatResp.Error.Message)
	}

	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("Chat API не вернул ни одного выбора")
	}

	return chatResp.Choices[0].Message.Content, nil
}
