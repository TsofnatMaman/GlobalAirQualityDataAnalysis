class CountryNotFoundException(Exception):

    def __init__(self, country_name):
        self._country_name = country_name

    def __str__(self):
        return "City {} not found".format(self._country_name)