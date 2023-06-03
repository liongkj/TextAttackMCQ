"""
word deletion Transformation
============================================

"""

from .transformation import Transformation


class WordDeletion(Transformation):
    """An abstract class that takes a sentence and transforms it by deleting a
    single word.

    letters_to_insert (string): letters allowed for insertion into words
    """

    def _get_transformations(self, current_text, indices_to_modify):
        # words = current_text.words
        transformed_texts = []
        if len(current_text.words) > 1:
            transformed_texts.extend(
                current_text.delete_word_at_index(i) for i in indices_to_modify
            )
        return transformed_texts
