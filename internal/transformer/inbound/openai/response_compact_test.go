package openai

import (
	"context"
	"testing"
)

func TestResponseInbound_TransformCompactRequest(t *testing.T) {
	inbound := &ResponseInbound{}
	ctx := context.WithValue(context.Background(), ResponseVariantContextKey{}, "compact")
	req, err := inbound.TransformRequest(ctx, []byte(`{"model":"octopus-codex","input":[],"stream":false}`))
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !req.IsResponseCompactRequest() {
		t.Fatalf("expected compact request metadata")
	}
	if req.Model != "octopus-codex" {
		t.Fatalf("expected model to be preserved, got %q", req.Model)
	}
	if string(req.RawRequest) != `{"model":"octopus-codex","input":[],"stream":false}` {
		t.Fatalf("expected raw request to be preserved")
	}
}

func TestResponseInbound_TransformCompactRequestRejectsStream(t *testing.T) {
	inbound := &ResponseInbound{}
	ctx := context.WithValue(context.Background(), ResponseVariantContextKey{}, "compact")
	_, err := inbound.TransformRequest(ctx, []byte(`{"model":"octopus-codex","input":[],"stream":true}`))
	if err == nil {
		t.Fatalf("expected stream compact request to fail")
	}
}
