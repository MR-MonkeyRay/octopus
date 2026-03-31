package openai

import (
	"context"
	"encoding/json"
	"io"
	"testing"

	"github.com/bestruirui/octopus/internal/transformer/model"
)

func TestResponseOutbound_TransformCompactRequest(t *testing.T) {
	outbound := &ResponseOutbound{}
	req, err := outbound.TransformRequest(context.Background(), &model.InternalLLMRequest{
		Model:      "gpt-5.1-codex-max",
		RawRequest: []byte(`{"model":"octopus-codex","input":[]}`),
		TransformerMetadata: map[string]string{
			"openai_responses_variant": "compact",
		},
	}, "https://example.com/v1", "sk-test")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if got := req.URL.String(); got != "https://example.com/v1/responses/compact" {
		t.Fatalf("unexpected request url: %s", got)
	}
	body, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("failed to read request body: %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("failed to unmarshal request body: %v", err)
	}
	if got, ok := payload["model"].(string); !ok || got != "gpt-5.1-codex-max" {
		t.Fatalf("unexpected model in request body: %#v", payload["model"])
	}
	if input, ok := payload["input"].([]any); !ok || len(input) != 0 {
		t.Fatalf("unexpected input in request body: %#v", payload["input"])
	}
	if got := req.Header.Get("Authorization"); got != "Bearer sk-test" {
		t.Fatalf("unexpected authorization header: %s", got)
	}
}

func TestResponseOutbound_TransformRequestRegularResponsesUnchanged(t *testing.T) {
	outbound := &ResponseOutbound{}
	req, err := outbound.TransformRequest(context.Background(), &model.InternalLLMRequest{
		Model: "gpt-5.1-codex-max",
		Messages: []model.Message{{
			Role: "user",
			Content: model.MessageContent{Content: compactStrPtr("hello")},
		}},
	}, "https://example.com/v1", "sk-test")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if got := req.URL.String(); got != "https://example.com/v1/responses" {
		t.Fatalf("unexpected request url: %s", got)
	}
}

func compactStrPtr(s string) *string { return &s }
