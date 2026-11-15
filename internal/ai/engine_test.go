package ai

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEngineConfiguration(t *testing.T) {
	e := NewEngine("", "")
	assert.Equal(t, ProviderGemini, e.Provider())
	assert.Equal(t, "gemini-2.0-flash", e.Model())
	assert.False(t, e.HasAPIKey())
	assert.Equal(t, "", e.MaskedAPIKey())

	// Configure Gemini with key
	err := e.Configure(ProviderGemini, "AIzaSyTestKey1234567890", "gemini-1.5-pro")
	assert.NoError(t, err)
	assert.Equal(t, ProviderGemini, e.Provider())
	assert.Equal(t, "gemini-1.5-pro", e.Model())
	assert.True(t, e.HasAPIKey())
	assert.Equal(t, "AIza••••7890", e.MaskedAPIKey())

	// Switch to OpenAI without changing key
	err = e.Configure(ProviderOpenAI, "", "")
	assert.NoError(t, err)
	assert.Equal(t, ProviderOpenAI, e.Provider())
	assert.Equal(t, "gemini-1.5-pro", e.Model()) // kept previous model since empty
	assert.True(t, e.HasAPIKey())

	// Switch model to gpt-4o and key to sk-test
	err = e.Configure(ProviderOpenAI, "sk-proj-1234567890abcdef", "gpt-4o")
	assert.NoError(t, err)
	assert.Equal(t, "gpt-4o", e.Model())
	assert.Equal(t, "sk-p••••cdef", e.MaskedAPIKey())

	// Test invalid provider
	err = e.Configure("unknown-provider", "", "")
	assert.Error(t, err)
}

func TestEngineChatCompleteWithoutKey(t *testing.T) {
	e := NewEngine("", "")
	_, err := e.ChatComplete(context.Background(), "sys", "user")
	assert.Error(t, err)

	_ = e.Configure(ProviderOpenAI, "", "gpt-4o")
	_, err = e.ChatComplete(context.Background(), "sys", "user")
	assert.Error(t, err)
}

func TestSupportedProviders(t *testing.T) {
	providers := SupportedProviders()
	assert.Len(t, providers, 2)
	assert.Equal(t, ProviderGemini, providers[0].ID)
	assert.Equal(t, ProviderOpenAI, providers[1].ID)

	// Ensure no invalid models like gemini-2.5-flash
	assert.NotContains(t, providers[0].Models, "gemini-2.5-flash")
	assert.Contains(t, providers[0].Models, "gemini-2.0-flash")
	assert.Contains(t, providers[0].Models, "gemini-2.0-flash-lite")
}

func TestFetchAvailableModelsWithoutKey(t *testing.T) {
	e := NewEngine("", "")
	_, err := e.FetchAvailableModels(context.Background(), ProviderGemini, "")
	assert.Error(t, err)

	_, err = e.FetchAvailableModels(context.Background(), ProviderOpenAI, "")
	assert.Error(t, err)
}

func TestSortGeminiModels(t *testing.T) {
	items := []ModelItem{
		{ID: "gemini-1.5-pro", DisplayName: "Gemini 1.5 Pro"},
		{ID: "gemini-2.0-flash", DisplayName: "Gemini 2.0 Flash"},
		{ID: "gemini-2.0-flash-lite", DisplayName: "Gemini 2.0 Flash Lite"},
		{ID: "gemini-1.5-flash", DisplayName: "Gemini 1.5 Flash"},
	}
	sortGeminiModels(items)
	assert.Equal(t, "gemini-2.0-flash", items[0].ID)
	assert.Equal(t, "gemini-2.0-flash-lite", items[1].ID)
	assert.Equal(t, "gemini-1.5-flash", items[2].ID)
	assert.Equal(t, "gemini-1.5-pro", items[3].ID)
}

