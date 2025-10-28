/**
 * DynamicTextureTemplate
 * ----------------------
 * Processing sketch template that exposes a rich set of parameters for
 * generating animated colour textures. Parameters are stored in a JSON file
 * and loaded through `loadParamsJSON()` so the same sketch can be reused with
 * different looks without recompilation.
 *
 * Usage tips
 * ----------
 * - Place a JSON file named `params.json` inside the sketch's `data/` folder.
 * - Press the `r` key while the sketch is running to reload parameters from disk.
 * - Press the `p` key to save a screenshot of the current frame to the sketch folder.
 */

// Path to the parameter JSON relative to the data folder.
String PARAMS_FILENAME = "params.json";

// Canvas controls (default values act as fallbacks if the JSON is missing fields).
int canvasWidth = 800;
int canvasHeight = 800;
int canvasSmooth = 4;
boolean useP2D = true;

// Animation timing parameters.
float time = 0.0;
float timeScale = 0.25f;
float noiseSpeed = 0.12f;
float driftSpeed = 0.35f;

// Texture characteristics.
float noiseScale = 0.008f;
float turbulence = 1.75f;
float lineFrequency = 3.2f;
float lineCurvature = 0.65f;
float textureContrast = 1.6f;
float textureBrightness = 1.05f;
float vignetteStrength = 0.35f;
float grainAmount = 0.05f;

// Palette controls.
color[] palette = {
  color(210, 80, 90, 100),
  color(320, 70, 85, 80),
  color(40, 90, 95, 90)
};

// Background blending controls.
float baseHue = 200;
float baseSaturation = 40;
float baseBrightness = 25;
float baseAlpha = 90;
float overlayHueShift = 35;
float overlayAlpha = 45;

// Internal buffers used for layered rendering.
PGraphics layerA;
PGraphics layerB;

void settings() {
  if (useP2D) {
    size(canvasWidth, canvasHeight, P2D);
  } else {
    size(canvasWidth, canvasHeight);
  }
  smooth(canvasSmooth);
}

void setup() {
  surface.setTitle("Dynamic Texture Template");
  colorMode(HSB, 360, 100, 100, 100);
  layerA = createGraphics(width, height, P2D);
  layerB = createGraphics(width, height, P2D);
  layerA.noSmooth();
  layerB.noSmooth();
  loadParamsJSON();
}

void draw() {
  float dt = max(1.0f / max(frameRate, 1), 1.0f / 60.0f);
  time += dt * timeScale;

  renderLayer(layerA, 0.0f, 0.75f);
  renderLayer(layerB, 0.45f, 1.0f);

  background(baseHue % 360, baseSaturation, baseBrightness, baseAlpha);
  blendMode(SCREEN);
  image(layerA, 0, 0);
  image(layerB, 0, 0);
  blendMode(BLEND);

  applyOverlay();
  applyVignette();
  applyGrain();
}

void renderLayer(PGraphics pg, float noiseOffset, float paletteOffset) {
  pg.beginDraw();
  pg.colorMode(HSB, 360, 100, 100, 100);
  pg.loadPixels();

  for (int y = 0; y < pg.height; y++) {
    float ny = y * noiseScale + noiseOffset;
    for (int x = 0; x < pg.width; x++) {
      float nx = x * noiseScale + noiseOffset;
      float n = 0;
      float scale = 1.0f;
      float amplitude = 1.0f;
      for (int octave = 0; octave < 4; octave++) {
        float angle = (x * lineFrequency / width) + (y * lineCurvature / height) + time * driftSpeed * amplitude;
        float stripe = sin(angle * TWO_PI);
        n += (noise(nx * scale, ny * scale, time * noiseSpeed * scale) * 0.6f + 0.4f * (stripe * 0.5f + 0.5f)) * amplitude;
        scale *= 2.0f;
        amplitude *= 0.5f;
      }
      n = pow(constrain(n / turbulence, 0, 1), textureContrast);
      n = applyBrightness(n, textureBrightness);

      color c1 = palette[(int)floor(((n + paletteOffset) % 1.0f) * palette.length)];
      color c2 = palette[(int)floor(((n + paletteOffset * 0.37f + 0.13f) % 1.0f) * palette.length)];
      color blended = lerpColor(c1, c2, abs(sin((n + paletteOffset) * PI)));
      blended = adjustAlpha(blended, 65 + n * 35);
      pg.pixels[y * pg.width + x] = blended;
    }
  }

  pg.updatePixels();
  pg.endDraw();
}

void applyOverlay() {
  pushStyle();
  noStroke();
  float overlayHue = (baseHue + overlayHueShift + sin(time * 0.75f) * 12.0f) % 360;
  float overlaySat = constrain(baseSaturation + 20 * sin(time * 1.1f), 0, 100);
  float overlayBright = constrain(baseBrightness + 30 * cos(time * 0.9f), 0, 100);
  fill(overlayHue, overlaySat, overlayBright, overlayAlpha);
  rect(0, 0, width, height);
  popStyle();
}

void applyVignette() {
  loadPixels();
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float dx = (x - width / 2.0f) / (width / 2.0f);
      float dy = (y - height / 2.0f) / (height / 2.0f);
      float dist = sqrt(dx * dx + dy * dy);
      float vignette = constrain(1.0f - pow(dist, 2.0f) * vignetteStrength, 0, 1);
      color current = pixels[y * width + x];
      float newAlpha = alpha(current) * vignette;
      pixels[y * width + x] = adjustAlpha(current, newAlpha);
    }
  }
  updatePixels();
}

void applyGrain() {
  loadPixels();
  for (int i = 0; i < pixels.length; i++) {
    float grain = (noise(i * 0.001f, time * 3.5f) - 0.5f) * grainAmount * 100;
    float h = hue(pixels[i]);
    float s = saturation(pixels[i]);
    float b = brightness(pixels[i]);
    b = constrain(b + grain, 0, 100);
    pixels[i] = color(h, s, b, alpha(pixels[i]));
  }
  updatePixels();
}

color adjustAlpha(color c, float newAlpha) {
  return color(hue(c), saturation(c), brightness(c), constrain(newAlpha, 0, 100));
}

float applyBrightness(float value, float factor) {
  if (factor > 1.0f) {
    return constrain(lerp(value, 1.0f, factor - 1.0f), 0, 1);
  } else if (factor < 1.0f) {
    return constrain(lerp(value, 0.0f, 1.0f - factor), 0, 1);
  }
  return constrain(value, 0, 1);
}

void keyPressed() {
  if (key == 'r' || key == 'R') {
    println("Reloading parameters from JSON ...");
    loadParamsJSON();
  } else if (key == 'p' || key == 'P') {
    String filename = "frame-" + year() + nf(month(), 2) + nf(day(), 2) + "_" + nf(hour(), 2) + nf(minute(), 2) + nf(second(), 2) + ".png";
    saveFrame(filename);
    println("Saved screenshot: " + filename);
  }
}

void loadParamsJSON() {
  try {
    JSONObject root = loadJSONObject(PARAMS_FILENAME);
    if (root == null) {
      println("Parameter file not found, using defaults.");
      return;
    }

    if (root.hasKey("canvas")) {
      JSONObject canvas = root.getJSONObject("canvas");
      if (canvas.hasKey("width")) canvasWidth = canvas.getInt("width");
      if (canvas.hasKey("height")) canvasHeight = canvas.getInt("height");
      if (canvas.hasKey("smooth")) canvasSmooth = canvas.getInt("smooth");
      if (canvas.hasKey("renderer")) useP2D = canvas.getString("renderer").equalsIgnoreCase("P2D");

      if (canvasWidth != width || canvasHeight != height) {
        surface.setSize(canvasWidth, canvasHeight);
        layerA = createGraphics(canvasWidth, canvasHeight, P2D);
        layerB = createGraphics(canvasWidth, canvasHeight, P2D);
      }
      smooth(canvasSmooth);
    }

    if (root.hasKey("animation")) {
      JSONObject animation = root.getJSONObject("animation");
      if (animation.hasKey("timeScale")) timeScale = animation.getFloat("timeScale");
      if (animation.hasKey("noiseSpeed")) noiseSpeed = animation.getFloat("noiseSpeed");
      if (animation.hasKey("driftSpeed")) driftSpeed = animation.getFloat("driftSpeed");
    }

    if (root.hasKey("texture")) {
      JSONObject texture = root.getJSONObject("texture");
      if (texture.hasKey("noiseScale")) noiseScale = texture.getFloat("noiseScale");
      if (texture.hasKey("turbulence")) turbulence = texture.getFloat("turbulence");
      if (texture.hasKey("lineFrequency")) lineFrequency = texture.getFloat("lineFrequency");
      if (texture.hasKey("lineCurvature")) lineCurvature = texture.getFloat("lineCurvature");
      if (texture.hasKey("contrast")) textureContrast = texture.getFloat("contrast");
      if (texture.hasKey("brightness")) textureBrightness = texture.getFloat("brightness");
      if (texture.hasKey("vignette")) vignetteStrength = texture.getFloat("vignette");
      if (texture.hasKey("grain")) grainAmount = texture.getFloat("grain");
    }

    if (root.hasKey("background")) {
      JSONObject background = root.getJSONObject("background");
      if (background.hasKey("hue")) baseHue = background.getFloat("hue");
      if (background.hasKey("saturation")) baseSaturation = background.getFloat("saturation");
      if (background.hasKey("brightness")) baseBrightness = background.getFloat("brightness");
      if (background.hasKey("alpha")) baseAlpha = background.getFloat("alpha");
    }

    if (root.hasKey("overlay")) {
      JSONObject overlay = root.getJSONObject("overlay");
      if (overlay.hasKey("hueShift")) overlayHueShift = overlay.getFloat("hueShift");
      if (overlay.hasKey("alpha")) overlayAlpha = overlay.getFloat("alpha");
    }

    if (root.hasKey("palette")) {
      JSONArray array = root.getJSONArray("palette");
      ArrayList<Integer> colors = new ArrayList<Integer>();
      for (int i = 0; i < array.size(); i++) {
        JSONObject entry = array.getJSONObject(i);
        float h = entry.hasKey("h") ? entry.getFloat("h") : random(0, 360);
        float s = entry.hasKey("s") ? entry.getFloat("s") : random(40, 100);
        float b = entry.hasKey("b") ? entry.getFloat("b") : random(40, 100);
        float a = entry.hasKey("a") ? entry.getFloat("a") : 100;
        colors.add(color(h % 360, constrain(s, 0, 100), constrain(b, 0, 100), constrain(a, 0, 100)));
      }
      if (!colors.isEmpty()) {
        palette = new color[colors.size()];
        for (int i = 0; i < colors.size(); i++) {
          palette[i] = colors.get(i);
        }
      }
    }

    println("Parameters loaded: " + PARAMS_FILENAME);
  } catch (RuntimeException ex) {
    println("Failed to load parameters: " + ex.getMessage());
  }
}
