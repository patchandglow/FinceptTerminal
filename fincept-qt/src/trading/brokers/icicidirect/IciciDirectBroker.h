#pragma once
#include "trading/BrokerInterface.h"
#include "trading/brokers/BrokerHttp.h"

namespace fincept::trading {

/// ICICI Direct (Breeze API) broker integration.
///
/// Auth: the user enters API Key + Secret Key, visits icici_login_url() to obtain
/// an "API Session" token, and pastes it back. exchange_token() trades that for a
/// base64 session token (via /customerdetails) which becomes the X-SessionToken
/// header. Every subsequent request is signed with
///   X-Checksum: token <SHA256(timestamp + json_body + secret_key)>
///
/// Breeze quirk: reads use GET-with-JSON-body, and one resource endpoint serves
/// multiple operations dispatched by HTTP verb (GET=read, POST=place, PUT=modify,
/// DELETE=cancel). Symbol mapping (normal ticker <-> ICICI stock_code, plus F&O
/// expiry/strike/right decomposition) comes from InstrumentService ("icicidirect").
///
/// Live tick streaming IS supported: IciciDirectWebSocket speaks Breeze's
/// Socket.IO (Engine.IO v4) transport and is constructed by AccountDataStream
/// for this broker id. It shipped in the same commit as the REST surface — the
/// "v1: REST only, streaming deferred" note that used to sit here was stale from
/// the first release and claimed the opposite of what the code does.
class IciciDirectBroker : public IBroker {
  public:
    BrokerId id() const override { return BrokerId::IciciDirect; }
    const char* name() const override { return "ICICI Direct"; }
    const char* base_url() const override { return "https://api.icicidirect.com/breezeapi/api/v1"; }
    // Nothing dispatches on this today — AccountDataStream picks the socket with
    // a hardcoded `broker_id_ == "..."` chain, and every other broker's adapter
    // name is likewise unread. Leaving it empty is still wrong: ICICI was the only
    // broker returning "" while all the others name their adapter, so the day this
    // accessor becomes the dispatch key, ICICI is the one that silently loses
    // streaming — and "" would read as a deliberate "no socket" rather than an
    // oversight.
    const char* ws_adapter_name() const override { return "icicidirect"; }

    BrokerProfile profile() const override {
        return BrokerProfile{
            .id = "icicidirect",
            .display_name = "ICICI Direct",
            .region = "IN",
            .currency = "INR",
            .credential_fields =
                {
                    {CredentialField::ApiKey, "API KEY", "Enter App Key (api_key)...", false},
                    {CredentialField::ApiSecret, "SECRET KEY", "Enter Secret Key...", true},
                    {CredentialField::AuthCode, "API SESSION", "Login via Breeze & paste API Session...", false},
                },
            .exchanges = {"NSE", "BSE", "NFO", "BFO", "MCX"},
            .product_types =
                {
                    {"Delivery (Cash)", ProductType::Delivery},
                    {"Intraday (Margin)", ProductType::Intraday},
                    {"Futures", ProductType::Margin},
                    {"MTF", ProductType::MTF},
                },
            .supports_intraday = true,
            .supports_bracket_order = false,
            .supports_cover_order = false,
            .has_native_paper = false,
            .default_paper_balance = 1000000.0,
            .default_watchlist = {"ICICIBANK", "HDFCBANK", "SBIN", "RELIANCE", "TCS", "INFY", "AXISBANK", "KOTAKBANK",
                                  "TATAMOTORS", "BAJFINANCE"},
            .default_symbol = "RELIANCE",
            .default_exchange = "NSE",
            .brokerage_info = "0.05% (cash)/0.05% or ₹20 (intraday); free API",
        };
    }

    /// URL the user visits to obtain an API Session token (pure string, any thread).
    static QString icici_login_url(const QString& api_key);

    TokenExchangeResponse exchange_token(const QString& api_key, const QString& api_secret,
                                         const QString& auth_code) override;
    OrderPlaceResponse place_order(const BrokerCredentials& creds, const UnifiedOrder& order) override;
    ApiResponse<QJsonObject> modify_order(const BrokerCredentials& creds, const QString& order_id,
                                          const QJsonObject& mods) override;
    ApiResponse<QJsonObject> cancel_order(const BrokerCredentials& creds, const QString& order_id) override;
    ApiResponse<QVector<BrokerOrderInfo>> get_orders(const BrokerCredentials& creds) override;
    ApiResponse<QJsonObject> get_trade_book(const BrokerCredentials& creds) override;
    ApiResponse<QVector<BrokerPosition>> get_positions(const BrokerCredentials& creds) override;
    ApiResponse<QVector<BrokerHolding>> get_holdings(const BrokerCredentials& creds) override;
    ApiResponse<BrokerFunds> get_funds(const BrokerCredentials& creds) override;
    ApiResponse<QVector<BrokerQuote>> get_quotes(const BrokerCredentials& creds,
                                                 const QVector<QString>& symbols) override;
    ApiResponse<QVector<BrokerCandle>> get_history(const BrokerCredentials& creds, const QString& symbol,
                                                   const QString& resolution, const QString& from_date,
                                                   const QString& to_date) override;

  protected:
    QMap<QString, QString> auth_headers(const BrokerCredentials& creds) const override;

  private:
    /// Fully-signed Breeze headers for a request whose exact serialised body is
    /// `body` (the checksum signs timestamp + body + secret_key).
    QMap<QString, QString> breeze_headers(const BrokerCredentials& creds, const QByteArray& body) const;
    /// Serialise `payload`, sign it, and dispatch `method` to base_url()+endpoint.
    BrokerHttpResponse breeze_request(const QString& method, const QString& endpoint, const QJsonObject& payload,
                                      const BrokerCredentials& creds);
    /// True when the Breeze envelope reports success ({"Status":200,"Error":null}).
    static bool is_success(const BrokerHttpResponse& resp);
    static QString checked_error(const BrokerHttpResponse& resp, const QString& fallback);
};

} // namespace fincept::trading
