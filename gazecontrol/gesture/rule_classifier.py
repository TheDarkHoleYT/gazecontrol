import gazecontrol.config as config


class RuleClassifier:
    def classify(self, features: dict) -> tuple:
        if features is None:
            return (None, 0.0)

        fs = features['finger_states']
        vy = features['hand_velocity_y']
        wrist_y = features['wrist_y']

        # PINCH: pollice e indice ravvicinati, altre dita estese
        # thumb_index_distance è normalizzato per la lunghezza della mano
        if (features['thumb_index_distance'] < 0.25
                and fs[2] == 1 and fs[3] == 1 and fs[4] == 1):
            return ('PINCH', 0.92)

        # SCROLL_UP: index+middle extended, others closed, upward velocity
        if (fs[1] == 1 and fs[2] == 1 and fs[0] == 0
                and fs[3] == 0 and fs[4] == 0
                and vy < -config.SWIPE_VELOCITY_THRESHOLD / 100):
            return ('SCROLL_UP', 0.90)

        # SCROLL_DOWN: index+middle extended, others closed, downward velocity
        if (fs[1] == 1 and fs[2] == 1 and fs[0] == 0
                and fs[3] == 0 and fs[4] == 0
                and vy > config.SWIPE_VELOCITY_THRESHOLD / 100):
            return ('SCROLL_DOWN', 0.90)

        # RELEASE: all fingers extended
        if fs == [1, 1, 1, 1, 1]:
            return ('RELEASE', 0.95)

        # CLOSE_SIGN: fist with thumb pointing down (tip below wrist in normalized coords)
        thumb_tip_y = features.get('_thumb_tip_y', None)
        if fs == [0, 0, 0, 0, 0]:
            # Use wrist_y and approximate thumb direction from angles
            # Thumb tip y > wrist y means thumb is below wrist (pointing down)
            if thumb_tip_y is not None and thumb_tip_y > wrist_y + 0.05:
                return ('CLOSE_SIGN', 0.90)
            return ('GRAB', 0.95)

        return (None, 0.0)
