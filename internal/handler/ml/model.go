package ml

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"prudo-detect/internal/domain"
)

type ModelAdapter struct {
	modelPath    string
	inferenceURL string // URL Python-сервиса с моделью
}

func NewModelAdapter(modelPath, inferenceURL string) *ModelAdapter {
	return &ModelAdapter{
		modelPath:    modelPath,
		inferenceURL: inferenceURL,
	}
}

// Predict выполняет inference через внешний Python-сервис
func (m *ModelAdapter) Predict(imageData []byte) ([]domain.BoundingBox, error) {
	// Создаём multipart запрос
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("file", "image.jpg")
	if err != nil {
		return nil, fmt.Errorf("create form file: %w", err)
	}

	if _, err := io.Copy(part, bytes.NewReader(imageData)); err != nil {
		return nil, fmt.Errorf("copy image data: %w", err)
	}

	writer.Close()

	// Отправляем запрос к Python-сервису
	req, err := http.NewRequest("POST", m.inferenceURL, body)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("inference failed with status: %d", resp.StatusCode)
	}

	// Парсим результат
	var result struct {
		Detections []domain.BoundingBox `json:"detections"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return result.Detections, nil
}

// CheckHealth проверяет доступность ML-сервиса
func (m *ModelAdapter) CheckHealth() error {
	resp, err := http.Get(m.inferenceURL + "/health")
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ml service unhealthy: %d", resp.StatusCode)
	}

	return nil
}
