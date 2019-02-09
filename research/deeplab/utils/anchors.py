import tensorflow as tf
class AnchorGenerator:
    def __init__(self, base_size, scales, ratios):
        self.base_size = base_size
        self.scales = tf.convert_to_tensor(scales, dtype=tf.float32)
        self.ratios = tf.convert_to_tensor(ratios, dtype=tf.float32)
        self.base_anchors = self._get_base_anchors()
    @property
    def num_base_anchors(self):
        return self.base_anchors
    
    def _get_base_anchors(self):
        base_anchor = tf.constant([0, 0, self.base_size - 1, self.base_size - 1], dtype=tf.float32)
        w = base_anchor[2] - base_anchor[0] + 1
        h = base_anchor[3] - base_anchor[1] + 1
        x_ctr = base_anchor[0] + 0.5 * (w - 1)
        y_ctr = base_anchor[1] + 0.5 * (h - 1)
        
        h_ratios = tf.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        
        # calculate width/height for each ratio
        ws = tf.reshape(w * w_ratios[:, None] * self.scales[None, :], [-1])
        hs = tf.reshape(h * h_ratios[:, None] * self.scales[None, :], [-1])
        
        base_anchors = tf.stack(
            [
                 x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ], axis=-1)
        
        # finally make sure you have rounded the output to integer
        return tf.cast(tf.math.round(base_anchors), dtype=tf.int32)
        
    def _mesh_grid(self, x, y):
        xx, yy = tf.meshgrid(x, y)
        xx, yy = tf.cast(xx, dtype=tf.int32), tf.cast(yy, dtype=tf.int32)
        return tf.reshape(xx, [-1]), tf.reshape(yy, [-1])
        
    def grid_anchors(self, featmap_size, stride=4):
        """Args
                featmap_size: (height, width)
            return: all anchors generated from the feature map: [N, 4]
        """
        base_anchors = self.base_anchors
        feat_w, feat_h = featmap_size 
        
        shift_x = tf.range(0, feat_w, delta=stride, dtype=tf.int32) 
        shift_y = tf.range(0, feat_h, delta=stride, dtype=tf.int32)
        
        shift_xx, shift_yy = self._mesh_grid(shift_x, shift_y)
        shifts = tf.stack([shift_xx, shift_yy, shift_xx, shift_yy], axis=-1)
        
        # add N anchors (1, N, 4) to K shifts (K, 1, 4) to get shifted anchors (N, A, 4), reshape to (N*A, 4)
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        # assume it is square shape
        all_anchors = tf.clip_by_value(all_anchors, 0, feat_w)
        
        return tf.reshape(all_anchors, [-1, 4])
    
        
        