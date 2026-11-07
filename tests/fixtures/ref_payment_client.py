import time, uuid
class PaymentGatewayClient:
    def __init__(self, sleep=time.sleep):
        self.sleep = sleep
    def new_audit_tag(self):
        return f"AUDIT_{uuid.uuid4()}"
    def retry_delay_ms(self, attempt):
        return (2 ** attempt) * 100
    def error_envelope(self, code, message):
        return {"error": {"code": code, "message": message, "audit_tag": self.new_audit_tag()}}
    def charge(self, amount_cents, send):
        last = None
        for attempt in range(3):
            try:
                return send(amount_cents)
            except ConnectionError as e:
                last = e
                if attempt < 2:
                    self.sleep(self.retry_delay_ms(attempt) / 1000)
        return self.error_envelope("GATEWAY_UNAVAILABLE", str(last))
