package tree_sitter_mlir_test

import (
	"testing"

	tree_sitter "github.com/smacker/go-tree-sitter"
	"github.com/tree-sitter/tree-sitter-mlir"
)

func TestCanLoadGrammar(t *testing.T) {
	language := tree_sitter.NewLanguage(tree_sitter_mlir.Language())
	if language == nil {
		t.Errorf("Error loading Mlir grammar")
	}
}
