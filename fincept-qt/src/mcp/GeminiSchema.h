#pragma once
// GeminiSchema.h — translate a JSON-Schema tool parameter object into the
// OpenAPI-3.0 subset that Gemini's `FunctionDeclaration.parameters` accepts.
//
// Gemini does NOT take arbitrary JSON Schema. Its `Schema` message is a fixed
// field set, and the JSON parser in front of it is STRICT: one unrecognised
// key anywhere in the tree fails the WHOLE generateContent request with
//
//     Invalid JSON payload received. Unknown name "additionalProperties" at
//     'tools[0].function_declarations[0].parameters'
//
// — which reads to the user as "tools don't work with Gemini" even though
// every other provider accepts the identical schema. The same is true of an
// OBJECT schema with an empty `properties` map:
//
//     GenerateContentRequest.tools[0].function_declarations[0].parameters.properties:
//     should be non-empty for OBJECT type
//
// …so a parameterless tool must omit `parameters` entirely rather than send
// `{"type":"object","properties":{}}` (what every OpenAI-shaped builder emits).
//
// Both failures are all-or-nothing: one bad declaration in the array takes the
// whole request down, so this translation is not best-effort polish — it is
// the difference between all tools working and none.

#include <QJsonObject>
#include <QString>

#include <optional>

namespace fincept::mcp {

/// Rewrite `schema` recursively, keeping only keys Gemini's Schema message
/// defines and normalising the values it is picky about (type casing, `format`
/// vocabulary, `enum` on non-string types).
///
/// Returns std::nullopt when the result carries nothing Gemini can accept —
/// an object with no usable properties. The caller MUST then omit the field
/// this schema belongs to: `parameters` at the top level, or the property
/// itself when nested.
std::optional<QJsonObject> sanitize_schema_for_gemini(const QJsonObject& schema);

/// Gemini's `FunctionDeclaration.name` constraint: must start with a letter or
/// underscore, may contain letters, digits, underscores, dots, dashes and
/// colons, and is capped at 64 characters. Internal tool names fit today
/// (longest is 56 including the `fincept-terminal__` prefix), but an external
/// MCP server with a long id can exceed it, and Gemini rejects the entire
/// request when it does.
bool is_valid_gemini_function_name(const QString& name);

/// Undo the STRING-surrogate substitution `sanitize_schema_for_gemini` applies
/// to free-form OBJECT parameters, so tool handlers keep receiving real JSON.
///
/// Gemini cannot declare "an object with arbitrary keys" — it has no
/// `additionalProperties`, and an OBJECT with an empty `properties` map is
/// rejected outright. Such a parameter therefore goes out as a STRING holding
/// serialised JSON, and the model fills it with a string. Left alone,
/// `args["params"].toObject()` in the handler yields `{}` — the call silently
/// does nothing, which is worse than an error.
///
/// Walks `original_schema` (the UNSANITISED tool schema) and, wherever it
/// declares an object or array but the model supplied a string that parses as
/// one, substitutes the parsed value. Anything that does not parse is left
/// untouched so the normal argument validator still reports it.
QJsonObject restore_gemini_call_args(const QJsonObject& original_schema, const QJsonObject& args);

} // namespace fincept::mcp
