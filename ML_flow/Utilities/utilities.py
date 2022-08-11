class Utilities:

    def __init__(self):
        '''
        Constructor classes
        '''
        
        self.dni_regex = "([a-z]|[A-Z]|[0-9])[0-9]{7}([a-z]|[A-Z]|[0-9])"
        self.email_regex = '^(.+\@.+\..+)$'
        self.movil_regex = '(\+34|0034|34)?[ -]*(6|7)[ -]*([0-9][ -]*){8}'
        self.telephone_regex = "(\+34|0034|34)?[ -]*(8|9)[ -]*([0-9][ -]*){8}"
        self.web_regex = '^(ht|f)tp(s?)\:\/\/[0-9a-zA-Z]([-.\w]*[0-9a-zA-Z])*(:(0-9)*)*(\/?)' \
                         '( [a-zA-Z0-9\-\.\?\,\’\/\\\+&amp;%\$#_]*)?$'
        self.datetime_regex = '^\d{2,5}[-/]\d{2}[-/]\d{2,5}$'
        self.val_cnae93 = list(pd.read_csv('../Input/CNAE93.csv')['Clase'])
        self.DIGITO_CONTROL = "TRWAGMYFPDXBNJZSQVHLCKE"