package main

import (
	"encoding/json"
	"log"
	"net/http"
)

type task struct {
	ID   int    `json:"id"`
	Name string `json:"name"`
	Done bool   `json:"done"`
}

type echoRequest struct {
	Message string `json:"message"`
}

type echoResponse struct {
	Original string `json:"original"`
	Length   int    `json:"length"`
}

var tasks = []task{
	{ID: 1, Name: "Learn Go structs", Done: true},
	{ID: 2, Name: "Build a small HTTP server", Done: true},
	{ID: 3, Name: "Try making a POST request", Done: false},
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", healthHandler)
	mux.HandleFunc("GET /api/tasks", tasksHandler)
	mux.HandleFunc("POST /api/echo", echoHandler)

	server := http.Server{
		Addr:    ":8080",
		Handler: logRequests(mux),
	}

	log.Println("server running on http://localhost:8080")
	log.Println("try GET /health, GET /api/tasks, POST /api/echo")

	if err := server.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status":  "ok",
		"message": "Go server is running",
	})
}

func tasksHandler(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, tasks)
}

func echoHandler(w http.ResponseWriter, r *http.Request) {
	var req echoRequest

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "request body must be valid JSON",
		})
		return
	}

	writeJSON(w, http.StatusOK, echoResponse{
		Original: req.Message,
		Length:   len(req.Message),
	})
}

func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	if err := json.NewEncoder(w).Encode(data); err != nil {
		http.Error(w, "failed to write response", http.StatusInternalServerError)
	}
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.Printf("%s %s", r.Method, r.URL.Path)
		next.ServeHTTP(w, r)
	})
}
