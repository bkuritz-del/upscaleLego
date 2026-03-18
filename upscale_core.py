class UpscaleSettings:
    def __init__(self, scale_factor, output_directory):
        self.scale_factor = scale_factor
        self.output_directory = output_directory
        
    def validate(self):
        if self.scale_factor <= 0:
            raise ValueError("Scale factor must be greater than 0")
        if not os.path.exists(self.output_directory):
            raise ValueError("Output directory does not exist")


def process_batch(input_images, upscale_settings):
    upscale_settings.validate()
    for image_path in input_images:
        # Load the image
        image = load_image(image_path)
        
        # Process image
        upscaled_image = upscale_image(image, upscale_settings.scale_factor)
        
        # Save the upscaled image
        output_path = os.path.join(upscale_settings.output_directory, os.path.basename(image_path))
        save_image(upscaled_image, output_path)