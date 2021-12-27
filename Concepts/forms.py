from logging import PlaceHolder
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.fields.simple import SubmitField
from wtforms.validators import DataRequired, Length

class SearchForm(FlaskForm):
    searchfield = StringField('Search')

    submit_btn = SubmitField('Search')