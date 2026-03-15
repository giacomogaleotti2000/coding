package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHealthHandler(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	res := httptest.NewRecorder()

	healthHandler(res, req)

	if res.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, res.Code)
	}
}

func TestTasksHandler(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/tasks", nil)
	res := httptest.NewRecorder()

	tasksHandler(res, req)

	if res.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, res.Code)
	}
}

func TestEchoHandler(t *testing.T) {
	body := bytes.NewBufferString(`{"message":"hello go"}`)
	req := httptest.NewRequest(http.MethodPost, "/api/echo", body)
	req.Header.Set("Content-Type", "application/json")
	res := httptest.NewRecorder()

	echoHandler(res, req)

	if res.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, res.Code)
	}
}
