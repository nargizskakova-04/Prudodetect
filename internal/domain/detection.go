package domain

// BoundingBox представляет координаты найденного объекта
type BoundingBox struct {
	X      int     `json:"x"`
	Y      int     `json:"y"`
	Width  int     `json:"width"`
	Height int     `json:"height"`
	Class  string  `json:"class"` // "qr", "signature", "stamp"
	Conf   float32 `json:"confidence"`
}

// DetectionResult результат детекции
type DetectionResult struct {
	Detections []BoundingBox `json:"detections"`
	ImageURL   string        `json:"image_url"`
	Success    bool          `json:"success"`
	Message    string        `json:"message,omitempty"`
}

// DetectionRequest запрос на детекцию
type DetectionRequest struct {
	ImageData []byte // PDF конвертируется в изображение
	Filename  string
}
