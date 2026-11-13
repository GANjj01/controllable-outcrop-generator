class ModelError(ValueError):
    """A field-addressed, fail-closed modeling error."""
    def __init__(self, field, code, message, details=None):
        self.field, self.code, self.message = field, code, message
        self.details = details or {}
        super().__init__(f"{field}: {message} [{code}]")


def invalid(field, message):
    raise ModelError(field, "invalid_contract", message)


def unresolved(field, message, **details):
    raise ModelError(field, "unresolved", message, details)
