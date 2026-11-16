"""
Content-Security-Policy header on every response (defence in depth against XSS).

The pages use no JavaScript and no external files, so the policy can be strict: only
inline styles (needed for the bar widths) and data: images are allowed. Even if some
uploaded text slipped through escaping, a <script> in it could not run.
"""
CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:; "
    "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
)


class ContentSecurityPolicyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if "Content-Security-Policy" not in response:
            response["Content-Security-Policy"] = CSP
        return response
