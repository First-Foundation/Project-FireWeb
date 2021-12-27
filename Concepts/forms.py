from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.fields.simple import SubmitField
from wtforms.validators import DataRequired, Length

class SearchForm(FlaskForm):
    searchfield = StringField('Search', validators=[DataRequired(), Length(min=2, max=30)])

    submit_btn = SubmitField('Search')