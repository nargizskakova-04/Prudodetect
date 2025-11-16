package service

import (
	"fmt"
	"prudo-detect/internal/domain"
	"prudo-detect/internal/handler/ml"
)

type DetectorService struct {
	model *ml.ModelAdapter
}

func NewDetectorService(model *ml.ModelAdapter) *DetectorService {
	return &DetectorService{
		model: model,
	}
}

// DetectObjects выполняет детекцию объектов на изображении
func (s *DetectorService) DetectObjects(req domain.DetectionRequest) (*domain.DetectionResult, error) {
	// Валидация
	if len(req.ImageData) == 0 {
		return &domain.DetectionResult{
			Success: false,
			Message: "No image data provided",
		}, nil
	}

	// Вызываем модель
	detections, err := s.model.Predict(req.ImageData)
	if err != nil {
		return nil, fmt.Errorf("model prediction failed: %w", err)
	}

	// Фильтруем результаты с низкой уверенностью
	filtered := make([]domain.BoundingBox, 0)
	for _, det := range detections {
		if det.Conf > 0.3 { // Порог уверенности
			filtered = append(filtered, det)
		}
	}

	return &domain.DetectionResult{
		Detections: filtered,
		Success:    true,
		Message:    fmt.Sprintf("Found %d objects", len(filtered)),
	}, nil
}
