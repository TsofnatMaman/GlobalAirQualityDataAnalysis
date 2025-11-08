class CountryNotFoundException(Exception):
    def __init__(self, country_name: str) -> None:
        self.country_name = country_name
        super().__init__(f"Country '{country_name}' not found")

    def __str__(self) -> str:
        return f"Country '{self.country_name}' not found"
