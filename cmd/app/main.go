package main

import (
	"log"
	"net/http"
	"prudo-detect/internal/config"
	httpHandler "prudo-detect/internal/handler/http"
	"prudo-detect/internal/handler/ml"
	"prudo-detect/internal/service"
)

func main() {
	// Загружаем конфигурацию
	cfg := config.Load()

	// Инициализируем ML-адаптер
	modelAdapter := ml.NewModelAdapter(cfg.ModelPath, cfg.InferenceURL)

	// Проверяем доступность ML-сервиса
	if err := modelAdapter.CheckHealth(); err != nil {
		log.Printf("Warning: ML service not available: %v", err)
	}

	// Создаём сервис
	detectorService := service.NewDetectorService(modelAdapter)

	// Создаём HTTP handler
	handler := httpHandler.NewHandler(detectorService)

	// Настраиваем роуты
	mux := http.NewServeMux()
	mux.HandleFunc("/predict", handler.PredictHandler)
	mux.HandleFunc("/health", handler.HealthHandler)

	// Отдаём статические файлы
	fs := http.FileServer(http.Dir("./static"))
	mux.Handle("/", fs)

	// Запускаем сервер
	addr := ":" + cfg.Port
	log.Printf("🚀 Server starting on http://localhost%s", addr)
	log.Printf("📊 ML inference URL: %s", cfg.InferenceURL)

	if err := http.ListenAndServe(addr, corsMiddleware(mux)); err != nil {
		log.Fatal(err)
	}
}

// corsMiddleware добавляет CORS заголовки
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}
