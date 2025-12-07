package ai

import (
	"context"
	"fmt"
	"os"
	"time"

	"google.golang.org/genai"
)

// GoogleProvider implements AIProvider using Google Gemini API
type GoogleProvider struct {
	client *genai.Client
	conf   ProviderConfig
}

// NewGoogleProvider creates a new Google Gemini provider
func NewGoogleProvider(client *genai.Client, conf ProviderConfig) *GoogleProvider {
	return &GoogleProvider{client: client, conf: conf}
}

func (g *GoogleProvider) generateWithRetry(ctx context.Context, contents []*genai.Content) (string, error) {
	var lastErr error

	// Try primary model
	for attempt := 1; attempt <= g.conf.PrimaryModelRetries; attempt++ {
		resp, err := g.client.Models.GenerateContent(ctx, g.conf.PrimaryModel, contents, nil)
		if err == nil {
			if txt := resp.Text(); txt != "" {
				return txt, nil
			}
			lastErr = fmt.Errorf("API вернул пустой текстовый ответ")
		} else {
			lastErr = err
		}
		if isRetryableError(lastErr) && attempt < g.conf.PrimaryModelRetries {
			time.Sleep(g.conf.RetryDelay)
			continue
		}
		break
	}

	// Try fallback model
	for attempt := 1; attempt <= g.conf.FallbackModelRetries; attempt++ {
		resp, err := g.client.Models.GenerateContent(ctx, g.conf.FallbackModel, contents, nil)
		if err == nil {
			if txt := resp.Text(); txt != "" {
				return txt, nil
			}
			lastErr = fmt.Errorf("API вернул пустой текстовый ответ")
		} else {
			lastErr = err
		}
		if isRetryableError(lastErr) && attempt < g.conf.FallbackModelRetries {
			time.Sleep(g.conf.RetryDelay)
			continue
		}
		break
	}

	return "", fmt.Errorf("все попытки генерации контента не удались, последняя ошибка: %w", lastErr)
}

// AudioToText transcribes audio file to text using Google Gemini
func (g *GoogleProvider) AudioToText(ctx context.Context, audioPath string) (string, error) {
	audioData, err := os.ReadFile(audioPath)
	if err != nil {
		return "", fmt.Errorf("не удалось прочитать аудиофайл: %w", err)
	}

	prompt := genai.NewPartFromText("Пожалуйста, транскрибируйте этот аудио файл в текст на том языке, на котором говорят в записи. Верните только текст транскрипции без дополнительных комментариев.")
	audioPart := genai.NewPartFromBytes(audioData, "audio/mpeg")
	contents := []*genai.Content{{Parts: []*genai.Part{prompt, audioPart}}}

	return g.generateWithRetry(ctx, contents)
}

// SummarizeText creates a summary using Google Gemini
func (g *GoogleProvider) SummarizeText(ctx context.Context, textToSummarize, promptTemplate string) (string, error) {
	userPrompt := fmt.Sprintf(promptTemplate, textToSummarize)
	contents := []*genai.Content{
		{Role: "user", Parts: []*genai.Part{genai.NewPartFromText(g.conf.SystemPrompt)}},
		{Role: "model", Parts: []*genai.Part{genai.NewPartFromText("Понял, буду следовать указанным правилам форматирования и структуры.")}},
		{Role: "user", Parts: []*genai.Part{genai.NewPartFromText(userPrompt)}},
	}
	return g.generateWithRetry(ctx, contents)
}
