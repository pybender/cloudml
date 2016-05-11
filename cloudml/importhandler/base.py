class AmazonSettingsMixin(object):
    def get_amazon_settings(self):
        if hasattr(self, "_amazon_settings"):
            return self._amazon_settings
        from config import AMAZON_ACCESS_TOKEN, \
            AMAZON_TOKEN_SECRET, BUCKET_NAME
        return (AMAZON_ACCESS_TOKEN, AMAZON_TOKEN_SECRET, BUCKET_NAME)

    def set_amazon_settings(self, value):
        if len(value) != 3:
            raise ValueError('need to specify token, secret and bucket')
        self._amazon_settings = value

    amazon_settings = property(get_amazon_settings, set_amazon_settings)
