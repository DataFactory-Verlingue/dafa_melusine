from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
from melusine.utils.transformer_scheduler import TransformerScheduler
from collections import Counter
from itertools import chain
import pandas as pd



class MetaNameSender(BaseEstimator, TransformerMixin):
    """Transformer which creates 'sender' feature extracted.
    It extracts the beginning of mail adresses.
    Compatible with scikit-learn API.
    """

    def __init__(self, topn_sender=100):
        self.le_sender = preprocessing.LabelEncoder()
        self.topn_sender = topn_sender

    def fit(self, X, y=None):

        if isinstance(X, dict):
            raise TypeError(
                "You should not use fit on a dictionary object. Use a DataFrame"
            )

        """ Fit LabelEncoder on encoded extensions."""
        X["sender"] = X.apply(self.get_sender, axis=1)
        self.top_sender = self.get_top_sender(X, n=self.topn_sender)
        X["sender"] = X.apply(
            self.encode_sender, args=(self.top_sender,), axis=1
        )
        # TODO : on ajoute other dans la liste des senders pour l'apprentissage du le pour que ca soit pris en compte par la suite
        self.le_sender.fit(pd.concat((X["sender"], pd.Series(["other"]))))
        return self

    def transform(self, X):
        """Encode sender"""

        if isinstance(X, dict):
            apply_func = TransformerScheduler.apply_dict
        else:
            apply_func = TransformerScheduler.apply_pandas

        X["sender"] = apply_func(X, self.get_sender)
        X["sender"] = apply_func(
            X, self.encode_sender, args_=(self.top_sender,)
        )
        
        if isinstance(X["sender"], str):
            X["sender"] = self.le_sender.transform([X["sender"]])[0]
        else:
            X["sender"] = self.le_sender.transform(X["sender"])
        return X

    @staticmethod
    def get_sender(row):
        """Gets the name of the sender from email address."""
        x = row["from"]
        try:
            if x == None:
                raise ValueError("Pas d'expéditeur dans le mail !")
            else:
                sender = x.split("@")[0].replace(".", "-")
        except Exception:
            return ""
        return sender

    @staticmethod
    def get_top_sender(X, n=100):
        "Returns list of most common name of senders."
        a = Counter(X["sender"].values)
        a = a.most_common(n)
        a = [x[0] for x in a]
        return a

    @staticmethod
    def encode_sender(row, top_sender):
        x = row["sender"]
        """Encode most common extensions and set the rest to 'other'."""
        if x in top_sender:
            return x
        else:
            return "other"





class MetaNameReceivers(BaseEstimator, TransformerMixin):
    """Transformer which creates 'receivers' feature extracted. 
    It extracts name of receivers in mail adress.
    Compatible with scikit-learn API.
    """

    def __init__(self, topn_receivers=100):
        self.le_receivers = preprocessing.LabelEncoder()
        self.topn_receivers = topn_receivers

    def fit(self, X, y=None):
        if isinstance(X, dict):
            raise TypeError(
                "You should not use fit on a dictionary object. Use a DataFrame"
            )
        """ Fit LabelEncoder on encoded receivers."""
        X["receivers"] = X.apply(self.get_receivers, axis=1)
        self.top_receivers = self.get_top_receivers(
            X, n=self.topn_receivers
        )
        X["receivers"] = X.apply(
            self.encode_receivers, args=(self.top_receivers,), axis=1
        )
        # TODO : on ajoute other dans la liste des receivers pour l'apprentissage du label encoder pour que ca soit pris en compte par la suite  
        self.le_receivers.fit(pd.concat((X["receivers"], pd.Series([['other']]))).sum())
        return self

    def transform(self, X):
        """Encode receivers names"""

        if isinstance(X, dict):
            apply_func = TransformerScheduler.apply_dict
        else:
            apply_func = TransformerScheduler.apply_pandas

        X["receivers"] = apply_func(X, self.get_receivers)
        X["receivers"] = apply_func(
            X, self.encode_receivers, args_=(self.top_receivers,)
        )
        if isinstance(X["receivers"], list):
            X["receivers"] = self.le_extension.transform([X["receivers"]])[
                0
            ]
        else:
            X["receivers"] = [
                self.le_receivers.transform(t) for t in X["receivers"]
            ]
        return X

    @staticmethod
    def get_receivers(row):
        """Gets receivers."""
        x = row["to"]
        receivers = []
        try:
            if x == None:
                raise ValueError("Pas de destinataires dans le mail !")
            else:
                if type(x) == str:
                    x = eval(x)
                for receiver in x:
                    receivers.append(receiver.split("@")[0].replace(".", "-"))            
        except Exception:
            return ""
        return receivers

    @staticmethod
    def get_top_receivers(X, n=100):
        "Returns list of most common receivers."
        receivers_counter = Counter(chain(*X["receivers"]))
        receivers_counter = receivers_counter.most_common(n)
        top_receivers = [x[0] for x in receivers_counter]
        return top_receivers

    @staticmethod
    def encode_receivers(row, top_receivers):
        x = row["receivers"]
        """Encode most common receivers and set the rest to 'other'."""
        encode = []
        if x:
            for rec in x:
                if rec in top_receivers:
                    encode.append(rec)
                else:
                    encode.append("other")
        else:  # No attachments
            encode.append("none")
        return encode
