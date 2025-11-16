package http

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"prudo-detect/internal/domain"
	"prudo-detect/internal/service"
)

type Handler struct {
	detector *service.DetectorService
}

func NewHandler(detector *service.DetectorService) *Handler {
	return &Handler{
		detector: detector,
	}
}

// PredictHandler обрабатывает POST /predict
func (h *Handler) PredictHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Парсим multipart form
	err := r.ParseMultipartForm(50 << 20) // 50MB max
	if err != nil {
		respondError(w, "Failed to parse form", http.StatusBadRequest)
		return
	}

	// Получаем файл
	file, header, err := r.FormFile("file")
	if err != nil {
		respondError(w, "No file uploaded", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Читаем содержимое
	imageData, err := io.ReadAll(file)
	if err != nil {
		respondError(w, "Failed to read file", http.StatusInternalServerError)
		return
	}

	// Вызываем сервис
	result, err := h.detector.DetectObjects(domain.DetectionRequest{
		ImageData: imageData,
		Filename:  header.Filename,
	})

	if err != nil {
		respondError(w, fmt.Sprintf("Detection failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Отправляем результат
	respondJSON(w, result, http.StatusOK)
}

// HealthHandler проверка здоровья сервиса
func (h *Handler) HealthHandler(w http.ResponseWriter, r *http.Request) {
	respondJSON(w, map[string]string{"status": "ok"}, http.StatusOK)
}

func respondJSON(w http.ResponseWriter, data interface{}, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func respondError(w http.ResponseWriter, message string, status int) {
	respondJSON(w, map[string]string{"error": message}, status)
}
