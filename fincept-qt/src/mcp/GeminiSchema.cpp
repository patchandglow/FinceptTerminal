// GeminiSchema.cpp — see GeminiSchema.h.

#include "mcp/GeminiSchema.h"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonValue>
#include <QRegularExpression>
#include <QSet>
#include <QString>

namespace fincept::mcp {

namespace {

// Field set of Gemini's `Schema` message. Anything outside this list is
// dropped rather than forwarded — the API's JSON parser rejects unknown names
// outright, and a dropped constraint only costs the model a hint whereas a
// forwarded one costs the entire request.
//
// Deliberately absent (and present in real JSON Schema): additionalProperties,
// $schema, $id, $ref, $defs, definitions, oneOf, allOf, not, const,
// exclusiveMinimum, exclusiveMaximum, multipleOf, uniqueItems, examples,
// contains, patternProperties, dependentRequired.
const QSet<QString>& gemini_allowed_schema_keys() {
    static const QSet<QString> kKeys = {
        QStringLiteral("type"),        QStringLiteral("format"),       QStringLiteral("title"),
        QStringLiteral("description"), QStringLiteral("nullable"),     QStringLiteral("enum"),
        QStringLiteral("items"),       QStringLiteral("properties"),   QStringLiteral("required"),
        QStringLiteral("minItems"),    QStringLiteral("maxItems"),     QStringLiteral("minProperties"),
        QStringLiteral("maxProperties"), QStringLiteral("minLength"),  QStringLiteral("maxLength"),
        QStringLiteral("pattern"),     QStringLiteral("minimum"),      QStringLiteral("maximum"),
        QStringLiteral("default"),     QStringLiteral("anyOf"),        QStringLiteral("example"),
        QStringLiteral("propertyOrdering"),
    };
    return kKeys;
}

// Gemini's Type enum, lower-cased. JSON Schema's "null" has no Type member —
// it is expressed as `nullable: true`, so a bare "null" type is not usable.
QString gemini_normalise_type(const QJsonValue& raw) {
    // A type UNION (`["string","null"]`) is legal JSON Schema and illegal here.
    // Take the first non-null member; nullability is carried separately.
    QString t;
    if (raw.isArray()) {
        for (const auto& v : raw.toArray()) {
            const QString s = v.toString().toLower();
            if (!s.isEmpty() && s != QLatin1String("null")) {
                t = s;
                break;
            }
        }
    } else {
        t = raw.toString().toLower();
    }
    static const QSet<QString> kTypes = {
        QStringLiteral("string"), QStringLiteral("number"), QStringLiteral("integer"),
        QStringLiteral("boolean"), QStringLiteral("array"), QStringLiteral("object"),
    };
    return kTypes.contains(t) ? t : QString();
}

// `format` is validated against a small per-type vocabulary; an unrecognised
// value ("uri", "email", "uuid", …) is a hard 400. Keep only what Gemini names.
bool gemini_format_is_valid(const QString& type, const QString& format) {
    if (type == QLatin1String("string")) {
        static const QSet<QString> kStr = {QStringLiteral("date-time"), QStringLiteral("date"),
                                           QStringLiteral("time"), QStringLiteral("duration"),
                                           QStringLiteral("enum")};
        return kStr.contains(format);
    }
    if (type == QLatin1String("integer")) {
        return format == QLatin1String("int32") || format == QLatin1String("int64");
    }
    if (type == QLatin1String("number")) {
        return format == QLatin1String("float") || format == QLatin1String("double");
    }
    return false;
}

std::optional<QJsonObject> gemini_sanitize(const QJsonObject& in, int depth);

// Sanitise the schema for one property. Returns nullopt when the property must
// be dropped (an object that ends up with no properties of its own).
std::optional<QJsonObject> gemini_sanitize_child(const QJsonValue& v, int depth) {
    if (!v.isObject())
        return std::nullopt;
    return gemini_sanitize(v.toObject(), depth);
}

std::optional<QJsonObject> gemini_sanitize(const QJsonObject& in, int depth) {
    // Guard against a self-referential schema from an external MCP server.
    // Gemini's own nesting ceiling is far below this.
    if (depth > 12)
        return std::nullopt;

    QJsonObject out;

    QString type = gemini_normalise_type(in.value(QStringLiteral("type")));
    // No usable type: infer object when it declares properties, else string —
    // a typeless leaf is more likely a free-text field than a broken object.
    if (type.isEmpty())
        type = in.contains(QStringLiteral("properties")) ? QStringLiteral("object") : QStringLiteral("string");
    out[QStringLiteral("type")] = type;

    // A `["string","null"]` union or an explicit nullable flag both map onto
    // Gemini's dedicated `nullable` field.
    const QJsonValue raw_type = in.value(QStringLiteral("type"));
    if (raw_type.isArray()) {
        for (const auto& v : raw_type.toArray()) {
            if (v.toString().toLower() == QLatin1String("null")) {
                out[QStringLiteral("nullable")] = true;
                break;
            }
        }
    }

    for (auto it = in.constBegin(); it != in.constEnd(); ++it) {
        const QString key = it.key();
        if (key == QLatin1String("type"))
            continue; // handled above
        if (!gemini_allowed_schema_keys().contains(key))
            continue;

        if (key == QLatin1String("format")) {
            const QString f = it.value().toString();
            if (gemini_format_is_valid(type, f))
                out[key] = f;
            continue;
        }

        if (key == QLatin1String("enum")) {
            // Gemini only accepts an enum on a STRING schema, and every member
            // must be a string. Numeric enums are expressed via minimum/maximum
            // instead; forwarding one is a 400.
            if (type != QLatin1String("string"))
                continue;
            QJsonArray vals;
            for (const auto& v : it.value().toArray()) {
                if (v.isString())
                    vals.append(v.toString());
            }
            if (!vals.isEmpty())
                out[key] = vals;
            continue;
        }

        if (key == QLatin1String("items")) {
            if (auto child = gemini_sanitize_child(it.value(), depth + 1))
                out[key] = *child;
            continue;
        }

        if (key == QLatin1String("anyOf")) {
            QJsonArray variants;
            for (const auto& v : it.value().toArray()) {
                if (auto child = gemini_sanitize_child(v, depth + 1))
                    variants.append(*child);
            }
            if (!variants.isEmpty())
                out[key] = variants;
            continue;
        }

        if (key == QLatin1String("properties")) {
            QJsonObject props;
            const QJsonObject src = it.value().toObject();
            for (auto pit = src.constBegin(); pit != src.constEnd(); ++pit) {
                if (auto child = gemini_sanitize_child(pit.value(), depth + 1))
                    props[pit.key()] = *child;
            }
            if (!props.isEmpty())
                out[key] = props;
            continue;
        }

        if (key == QLatin1String("required"))
            continue; // filtered below, once `properties` is final

        out[key] = it.value();
    }

    if (type == QLatin1String("array") && !out.contains(QStringLiteral("items"))) {
        // ARRAY without `items` is rejected. A string element is the least
        // surprising default and keeps the surrounding declaration usable.
        out[QStringLiteral("items")] = QJsonObject{{QStringLiteral("type"), QStringLiteral("string")}};
    }

    if (type == QLatin1String("object")) {
        const QJsonObject props = out.value(QStringLiteral("properties")).toObject();
        // An OBJECT with no properties is rejected outright ("should be
        // non-empty for OBJECT type"), and Gemini has no `additionalProperties`
        // with which to say "any keys". So a FREE-FORM object parameter — the
        // `params` / `config` / `args` / `metadata` bag that ~150 tools in this
        // catalogue declare — cannot be expressed as an OBJECT at all.
        //
        // Dropping it (what this used to do) is not a safe fallback: the tool is
        // then declared to Gemini as taking no such argument, so the model can
        // never supply it. For the ~115 tools whose ONLY parameter is such a
        // bag, the whole `parameters` object collapsed and the tool was
        // advertised as parameterless — callable, and useless.
        //
        // Nested, we substitute a STRING carrying serialised JSON, which is the
        // one shape Gemini both accepts and lets the model fill with arbitrary
        // keys. `restore_gemini_call_args` parses it back before dispatch, so
        // the handler still receives a real object.
        //
        // At the top level (depth 0) there is nothing to substitute into:
        // `parameters` itself must be an OBJECT, so the caller must omit the
        // field entirely — which is correct there, since a top-level empty
        // object means a genuinely parameterless tool.
        if (props.isEmpty()) {
            if (depth == 0)
                return std::nullopt;
            QString desc = in.value(QStringLiteral("description")).toString().trimmed();
            if (!desc.isEmpty() && !desc.endsWith(QLatin1Char('.')))
                desc += QLatin1Char('.');
            desc += QStringLiteral(" Pass a JSON object serialised as a string, e.g. "
                                   "\"{\\\"key\\\": \\\"value\\\"}\".");
            return QJsonObject{{QStringLiteral("type"), QStringLiteral("string")},
                               {QStringLiteral("description"), desc.trimmed()}};
        }

        // `required` may only name properties that survived sanitisation —
        // a required-but-absent name is itself a validation error.
        QJsonArray req;
        for (const auto& r : in.value(QStringLiteral("required")).toArray()) {
            const QString rn = r.toString();
            if (props.contains(rn))
                req.append(rn);
        }
        if (!req.isEmpty())
            out[QStringLiteral("required")] = req;
    }

    return out;
}

} // namespace

std::optional<QJsonObject> sanitize_schema_for_gemini(const QJsonObject& schema) {
    if (schema.isEmpty())
        return std::nullopt;
    return gemini_sanitize(schema, 0);
}

namespace {

// One value, coerced against the type its schema declares. `depth` guards a
// self-referential schema the same way the sanitiser does.
QJsonValue gemini_restore_value(const QJsonValue& value, const QJsonObject& schema, int depth);

QJsonObject gemini_restore_object(const QJsonObject& args, const QJsonObject& schema, int depth) {
    const QJsonObject props = schema.value(QStringLiteral("properties")).toObject();
    if (props.isEmpty() || depth > 12)
        return args;
    QJsonObject out = args;
    for (auto it = args.constBegin(); it != args.constEnd(); ++it) {
        const QJsonValue decl = props.value(it.key());
        if (!decl.isObject())
            continue;
        out[it.key()] = gemini_restore_value(it.value(), decl.toObject(), depth + 1);
    }
    return out;
}

QJsonValue gemini_restore_value(const QJsonValue& value, const QJsonObject& schema, int depth) {
    if (depth > 12)
        return value;
    const QString type = gemini_normalise_type(schema.value(QStringLiteral("type")));

    if (type == QLatin1String("object")) {
        if (value.isObject())
            return gemini_restore_object(value.toObject(), schema, depth);
        if (value.isString()) {
            // The surrogate round-trip. An empty string means "not supplied" —
            // returning it unchanged lets the validator speak for itself.
            const QJsonDocument doc = QJsonDocument::fromJson(value.toString().toUtf8());
            if (doc.isObject())
                return gemini_restore_object(doc.object(), schema, depth);
        }
        return value;
    }

    if (type == QLatin1String("array")) {
        QJsonArray arr;
        if (value.isArray()) {
            arr = value.toArray();
        } else if (value.isString()) {
            const QJsonDocument doc = QJsonDocument::fromJson(value.toString().toUtf8());
            if (!doc.isArray())
                return value;
            arr = doc.array();
        } else {
            return value;
        }
        const QJsonValue items = schema.value(QStringLiteral("items"));
        if (!items.isObject())
            return arr;
        QJsonArray out;
        for (const auto& e : arr)
            out.append(gemini_restore_value(e, items.toObject(), depth + 1));
        return out;
    }

    return value;
}

} // namespace

QJsonObject restore_gemini_call_args(const QJsonObject& original_schema, const QJsonObject& args) {
    if (original_schema.isEmpty() || args.isEmpty())
        return args;
    return gemini_restore_object(args, original_schema, 0);
}

bool is_valid_gemini_function_name(const QString& name) {
    static const QRegularExpression rx(QStringLiteral("^[a-zA-Z_][a-zA-Z0-9_.:-]{0,63}$"));
    return rx.match(name).hasMatch();
}

} // namespace fincept::mcp
