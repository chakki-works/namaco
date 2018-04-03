"""
Model API.
"""
import numpy as np
from seqeval.metrics.sequence_labeling import get_entities


mapping = {'corporation_other': 'organization', 'class': 'artifact', 'conference': 'event',
           'address_other': 'location', 'bridge': 'facility', 'time': 'timex',
           'event_other': 'event', 'period_week': 'timex', 'country': 'location',
           'drug': 'artifact', 'bird': 'natural_object', 'facility_other': 'facility',
           'goe_other': 'facility', 'god': 'name', 'frequency': 'quantity', 'mountain': 'location',
           'era': 'timex', 'weight': 'quantity', 'countx_other': 'quantity', 'period_date': 'timex',
           'n_country': 'quantity', 'n_product': 'quantity', 'period_year': 'timex',
           'language_other': 'artifact', 'intensity': 'quantity', 'canal': 'facility',
           'national_language': 'artifact', 'research_institute': 'facility', 'stock': 'quantity',
           'facility_part': 'facility', 'magazine': 'artifact', 'fungus': 'natural_object',
           'academic': 'artifact', 'company': 'organization', 'space': 'quantity',
           'calorie': 'quantity', 'political_party': 'organization', 'numex_other': 'quantity',
           'show': 'artifact', 'spa': 'location', 'river': 'location', 'dish': 'artifact',
           'music': 'artifact', 'incident_other': 'event', 'phone_number': 'location', 'age': 'quantity',
           'mollusc_anthropod': 'natural_object', 'movie': 'artifact', 'amphibia': 'natural_object',
           'doctrine_method_other': 'artifact', 'weapon': 'artifact', 'tunnel': 'facility',
           'element': 'natural_object', 'international_organization': 'organization',
           'religious_festival': 'event', 'theater': 'facility', 'sea': 'location', 'war': 'event',
           'car_stop': 'facility', 'periodx_other': 'timex', 'money_form': 'artifact',
           'public_institution': 'facility', 'fish': 'natural_object', 'percent': 'quantity',
           'sports_league': 'organization', 'continental_region': 'location', 'flora': 'natural_object',
           'living_thing_other': 'natural_object', 'date': 'timex', 'museum': 'facility',
           'picture': 'artifact', 'day_of_week': 'timex', 'book': 'artifact', 'n_event': 'quantity',
           'constellation': 'location', 'plan': 'artifact', 'sports_facility': 'facility',
           'location_other': 'location', 'natural_disaster': 'event', 'n_facility': 'quantity',
           'province': 'location', 'repitile': 'natural_object', 'nature_color': 'color',
           'treaty': 'artifact', 'currency': 'artifact', 'astral_body_other': 'location',
           'show_organization': 'organization', 'earthquake': 'event', 'flora_part': 'natural_object',
           'url': 'location', 'bay': 'location', 'newspaper': 'artifact', 'planet': 'location',
           'ethnic_group_other': 'organization', 'train': 'artifact', 'park': 'facility',
           'title_other': 'artifact', 'natural_object_other': 'natural_object', 'temperature': 'quantity',
           'cabinet': 'organization', 'food_other': 'artifact', 'period_time': 'timex',
           'nationality': 'organization', 'name_other': 'name', 'road': 'facility', 'city': 'location',
           'political_organization_other': 'organization', 'color_other': 'color',
           'sports_organization_other': 'organization', 'market': 'facility', 'period_month': 'timex',
           'spaceship': 'artifact', 'geological_region_other': 'location', 'n_animal': 'quantity',
           'car': 'artifact', 'worship_place': 'facility', 'military': 'organization',
           'tumulus': 'facility', 'offense': 'artifact', 'insect': 'natural_object',
           'aircraft': 'artifact', 'style': 'artifact', 'n_location_other': 'quantity',
           'multiplication': 'quantity', 'water_root': 'facility', 'person': 'name',
           'latitude_longtitude': 'quantity', 'occasion_other': 'event', 'company_group': 'organization',
           'star': 'location', 'culture': 'artifact', 'ship': 'artifact', 'id_number': 'artifact',
           'volume': 'quantity', 'pro_sports_organization': 'organization', 'region_other': 'location',
           'seismic_magnitude': 'quantity', 'county': 'location', 'vehicle_other': 'artifact',
           'religion': 'artifact', 'school': 'facility', 'postal_address': 'location',
           'rank': 'quantity', 'animal_part': 'natural_object', 'award': 'artifact',
           'school_age': 'quantity', 'n_flora': 'quantity', 'printing_other': 'artifact',
           'amusement_park': 'facility', 'ordinal_number': 'quantity', 'gpe_other': 'location',
           'measurement_other': 'quantity', 'sport': 'artifact', 'material': 'artifact',
           'airport': 'facility', 'living_thing_part_other': 'natural_object', 'clothing': 'artifact',
           'character': 'name', 'family': 'organization', 'position_vocation': 'artifact',
           'archaeological_place_other': 'facility', 'natural_phenomenon_other': 'event',
           'broadcast_program': 'artifact', 'games': 'event', 'point': 'quantity',
           'island': 'location', 'station': 'facility', 'product_other': 'artifact',
           'unit_other': 'artifact', 'n_natural_object_other': 'quantity', 'theory': 'artifact',
           'rule_other': 'artifact', 'physical_extent': 'quantity', 'government': 'organization',
           'seismic_intensity': 'quantity', 'service': 'artifact', 'organization_other': 'organization',
           'mineral': 'natural_object', 'n_organization': 'quantity', 'mammal': 'natural_object',
           'port': 'facility', 'animal_disease': 'disease', 'speed': 'quantity', 'art_other': 'artifact',
           'domestic_region': 'location', 'compound': 'natural_object', 'law': 'artifact',
           'railroad': 'facility', 'lake': 'location', 'n_person': 'quantity', 'line_other': 'facility'}


class Tagger(object):

    def __init__(self, model, preprocessor=None,
                 dynamic_preprocessor=None, tokenizer=str.split):
        self.model = model
        self.preprocessor = preprocessor
        self.dynamic_preprocessor = dynamic_preprocessor
        self.tokenizer = tokenizer

    def predict(self, sent):
        """Predict using the model.
        Args:
            sent : string, the input data.
       Returns:
           y : array-like, shape (n_samples,) or (n_samples, n_classes)
           The predicted classes.
       """
        X = self.preprocessor.transform([sent])
        X = self.dynamic_preprocessor.transform(X)
        y = self.model.predict(X)

        return y

    def _get_tags(self, pred):
        pred = np.argmax(pred, -1)
        tags = self.preprocessor.inverse_transform(docs=pred)

        return tags

    def _get_prob(self, pred):
        prob = np.max(pred, -1)[0]

        return prob

    def _build_response(self, sent, tags, prob):
        res = {
            'text': sent,
            'entities': [

            ]
        }
        chunks = get_entities(tags)

        for chunk_type, chunk_start, chunk_end in chunks:
            chunk_end += 1
            entity = {
                'text': sent[chunk_start: chunk_end],
                'type': mapping[chunk_type],
                'score': float(np.average(prob[chunk_start: chunk_end])),
                'beginOffset': chunk_start,
                'endOffset': chunk_end,
                'disambiguation': {
                    'sub_type': chunk_type,
                    'page_url': '',
                    'img_url': ''
                }
            }
            res['entities'].append(entity)

        return res

    def analyze(self, sent):
        assert isinstance(sent, str)

        pred = self.predict(sent)
        tags = self._get_tags(pred)
        prob = self._get_prob(pred)
        res = self._build_response(sent, tags, prob)

        return res

    def label(self, sent):
        pred = self.predict(sent)
        tags = self._get_tags(pred)

        return tags[0]
