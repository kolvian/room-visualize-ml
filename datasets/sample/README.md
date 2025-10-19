# Sample Images

This directory contains sample images for quick testing and demos.

## Style Images (`styles/`)

Famous artworks to use as style references:

- **starry_night.jpg** - Vincent van Gogh's "The Starry Night" (1889)
  - Best for: Swirling, expressive brushstrokes
  - Style: Post-Impressionism
  
- **wave.jpg** - Katsushika Hokusai's "The Great Wave off Kanagawa" (1831)
  - Best for: Bold patterns and Japanese aesthetics
  - Style: Ukiyo-e woodblock print
  
- **picasso.jpg** - Pablo Picasso's "Guernica" (1937)
  - Best for: Cubist, monochromatic abstraction
  - Style: Cubism
  
- **mosaic.jpg** - Byzantine mosaic pattern
  - Best for: Geometric, tiled effects
  - Style: Ancient mosaic art

## Content Images (`content/`)

Sample photos for testing style transfer:

- **city.jpg** - Times Square, New York City
  - Good for: Urban scenes, buildings, lights
  
- **landscape.jpg** - Mount Everest
  - Good for: Natural landscapes, mountains, outdoor scenes

## Quick Test

Run a 30-second test to see style transfer in action:

```bash
# Test with Starry Night style
python training/pytorch/train.py \
  --content-dir datasets/sample/content \
  --style-image datasets/sample/styles/starry_night.jpg \
  --style-name starry-night-test \
  --epochs 2 \
  --batch-size 1 \
  --checkpoint-dir checkpoints/test
```

## Adding Your Own Images

### Style Images
Add any artistic image (paintings, patterns, textures):
```bash
cp your_style.jpg datasets/sample/styles/
```

**Tips:**
- Use high-quality images (at least 512x512px)
- Strong artistic features work best (bold colors, clear brushstrokes)
- Avoid photos - use artwork, paintings, or stylized images

### Content Images
Add any photos you want to stylize:
```bash
cp your_photo.jpg datasets/sample/content/
```

**Tips:**
- Standard photos work great
- Good resolution (512x512 or larger)
- Clear subjects produce better results

## Image Sources

All sample images are from Wikimedia Commons under public domain or CC licenses:
- Starry Night: Public Domain (pre-1928)
- Great Wave: Public Domain (pre-1928)
- Guernica: Fair Use (educational/transformative)
- Mosaic: CC BY-SA 3.0
- City/Landscape: CC BY-SA 3.0

## License Note

When using these samples in production:
- The **trained models** you create are yours
- The **sample images** have their own licenses (see above)
- For commercial use, replace with your own licensed images
