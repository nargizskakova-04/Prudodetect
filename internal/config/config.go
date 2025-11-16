package config

import "os"

type Config struct {
	Port         string
	ModelPath    string
	InferenceURL string
}

func Load() *Config {
	return &Config{
		Port:         getEnv("PORT", "8080"),
		ModelPath:    getEnv("MODEL_PATH", "./models/best.pt"),
		InferenceURL: getEnv("INFERENCE_URL", "http://localhost:5000/predict"),
	}
}

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}
