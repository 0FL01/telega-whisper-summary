package main

import (
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/0fl01/voice-shut-up-bot-go/internal/ai"
	"github.com/0fl01/voice-shut-up-bot-go/internal/bot"
	"github.com/0fl01/voice-shut-up-bot-go/internal/config"
	"github.com/0fl01/voice-shut-up-bot-go/internal/media"
	"github.com/0fl01/voice-shut-up-bot-go/internal/telegram"
)

func main() {
	log.Println("Запуск бота...")

	cfg := config.LoadFromEnv()

	// Validate required environment variables
	if cfg.BotToken == "" {
		log.Fatalf("Переменная окружения %s должна быть установлена", config.EnvBotToken)
	}

	// Validate provider-specific variables
	switch strings.ToLower(cfg.AIProvider) {
	case config.ProviderGoogle:
		if cfg.GoogleAPIKey == "" {
			log.Fatalf("Для провайдера 'google' переменная окружения %s должна быть установлена", config.EnvGoogleAPIKey)
		}
		log.Printf("Используется провайдер: Google Gemini (модели: %s, %s)", cfg.PrimaryModel, cfg.FallbackModel)
	case config.ProviderGroq:
		if cfg.GroqAPIKey == "" {
			log.Fatalf("Для провайдера 'groq' переменная окружения %s должна быть установлена", config.EnvGroqAPIKey)
		}
		log.Printf("Используется провайдер: Groq (Whisper: %s, LLM: %s, Fallback: %s)",
			cfg.GroqWhisperModel, cfg.GroqPrimaryModel, cfg.GroqFallbackModel)
	default:
		log.Fatalf("Неизвестный провайдер: %s (допустимые значения: %s, %s)",
			cfg.AIProvider, config.ProviderGoogle, config.ProviderGroq)
	}

	apiBaseURL := fmt.Sprintf("https://api.telegram.org/bot%s", cfg.BotToken)
	httpClient := &http.Client{Timeout: 120 * time.Second}

	// Create AI provider using factory
	aiProvider, err := ai.NewProvider(cfg, httpClient)
	if err != nil {
		log.Fatalf("Не удалось создать AI провайдер: %v", err)
	}

	tele := telegram.NewClient(cfg.BotToken, apiBaseURL, httpClient)
	mediaProc := media.NewProcessor()

	application := bot.NewApp(cfg, tele, aiProvider, mediaProc)
	log.Println("Бот успешно запущен и готов к работе.")
	application.PollUpdates()
}
