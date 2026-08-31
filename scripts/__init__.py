"""Diagnostic and fitting scripts.

This file must exist. Without it `scripts` is only a NAMESPACE package, and a
namespace portion loses to any regular `scripts` package found later on
sys.path -- several environments ship one. The import then resolves to that
package, and every `from scripts.X import Y` in here dies with
`ModuleNotFoundError: No module named 'scripts.X'` even though X sits right
next to the importing file. That silently broke the entire diagnostic suite on
one machine while working on another.
"""
