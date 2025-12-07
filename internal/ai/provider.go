package ai

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/0fl01/voice-shut-up-bot-go/internal/config"
	"google.golang.org/genai"
)

// AIProvider defines the interface for AI services (transcription and summarization)
type AIProvider interface {
	// AudioToText transcribes audio file to text
	AudioToText(ctx context.Context, audioPath string) (string, error)
	// SummarizeText creates a summary of the given text using the provided prompt template
	SummarizeText(ctx context.Context, text, promptTemplate string) (string, error)
}

// ProviderConfig holds common configuration for AI providers
type ProviderConfig struct {
	PrimaryModel        string
	FallbackModel       string
	SystemPrompt        string
	UserPromptTemplate  string
	ShortPromptTemplate string

	PrimaryModelRetries  int
	FallbackModelRetries int
	RetryDelay           time.Duration
}

// NewProvider creates an AI provider based on the configuration
func NewProvider(cfg config.Config, httpClient *http.Client) (AIProvider, error) {
	switch strings.ToLower(cfg.AIProvider) {
	case config.ProviderGoogle:
		return newGoogleProvider(cfg)
	case config.ProviderGroq:
		return newGroqProvider(cfg, httpClient)
	default:
		return nil, fmt.Errorf("неизвестный AI провайдер: %s (допустимые значения: %s, %s)",
			cfg.AIProvider, config.ProviderGoogle, config.ProviderGroq)
	}
}

func newGoogleProvider(cfg config.Config) (*GoogleProvider, error) {
	if cfg.GoogleAPIKey == "" {
		return nil, fmt.Errorf("переменная окружения %s не установлена", config.EnvGoogleAPIKey)
	}

	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{APIKey: cfg.GoogleAPIKey})
	if err != nil {
		return nil, fmt.Errorf("не удалось создать клиент Google Gemini: %w", err)
	}

	return NewGoogleProvider(client, ProviderConfig{
		PrimaryModel:         cfg.PrimaryModel,
		FallbackModel:        cfg.FallbackModel,
		SystemPrompt:         cfg.SystemPrompt,
		UserPromptTemplate:   cfg.UserPromptTemplate,
		ShortPromptTemplate:  cfg.ShortPromptTemplate,
		PrimaryModelRetries:  cfg.PrimaryModelRetries,
		FallbackModelRetries: cfg.FallbackModelRetries,
		RetryDelay:           cfg.RetryDelay,
	}), nil
}

func newGroqProvider(cfg config.Config, httpClient *http.Client) (*GroqProvider, error) {
	if cfg.GroqAPIKey == "" {
		return nil, fmt.Errorf("переменная окружения %s не установлена", config.EnvGroqAPIKey)
	}

	if httpClient == nil {
		httpClient = &http.Client{Timeout: 120 * time.Second}
	}

	return NewGroqProvider(httpClient, cfg.GroqAPIKey, ProviderConfig{
		PrimaryModel:         cfg.GroqPrimaryModel,
		FallbackModel:        cfg.GroqFallbackModel,
		SystemPrompt:         cfg.SystemPrompt,
		UserPromptTemplate:   cfg.UserPromptTemplate,
		ShortPromptTemplate:  cfg.ShortPromptTemplate,
		PrimaryModelRetries:  cfg.PrimaryModelRetries,
		FallbackModelRetries: cfg.FallbackModelRetries,
		RetryDelay:           cfg.RetryDelay,
	}, cfg.GroqWhisperModel), nil
}

// isRetryableError checks if an error is retryable (common for all providers)
func isRetryableError(err error) bool {
	if err == nil {
		return false
	}
	es := strings.ToLower(err.Error())
	retryable := []string{"503", "429", "500", "overloaded", "unavailable", "timeout", "deadline exceeded", "rate limit"}
	for _, s := range retryable {
		if strings.Contains(es, s) {
			return true
		}
	}
	return false
}
