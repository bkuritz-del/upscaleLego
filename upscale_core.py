    def rgb_to_cmyk(rgb: tuple[int, int, int]) -> tuple[int, int, int, int]:
        """
        Convert an RGB color tuple to CMYK.
        
        Args:
            rgb: A tuple of (R, G, B) values, each 0-255
            
        Returns:
            A tuple of (C, M, Y, K) values, each 0-255
        """
        r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
        
        k = 1.0 - max(r, g, b)
        
        if k == 1.0:
            return (0, 0, 0, 255)
        
        c = (1.0 - r - k) / (1.0 - k)
        m = (1.0 - g - k) / (1.0 - k)
        y = (1.0 - b - k) / (1.0 - k)
        
        return (
            int(round(c * 255)),
            int(round(m * 255)),
            int(round(y * 255)),
            int(round(k * 255)),
        )
