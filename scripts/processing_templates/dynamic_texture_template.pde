// Dynamic Texture Template for Processing
//
// This sketch demonstrates how to create a multi-layered, colour-rich texture
// using animated Perlin noise. Parameters are loaded from a JSON file via the
// loadParamsJSON() helper so that designers can iterate without recompiling.
// Press 'R' to reload the JSON file at runtime.

SketchParams params = new SketchParams();
String PARAM_FILE = "data/dynamic_texture_params.json";
boolean sketchReady = false;

void settings() {
  loadParamsJSON();
  size(params.canvasWidth, params.canvasHeight, P2D);
}

void setup() {
  colorMode(HSB, 360, 100, 100, 100);
  surface.setResizable(true);
  applyRuntimeParams();
  sketchReady = true;
}

void draw() {
  background(
    params.backgroundHue,
    params.backgroundSaturation,
    params.backgroundBrightness,
    params.backgroundAlpha
  );

  float timeSeconds = millis() * 0.001;
  float paletteTime = timeSeconds * params.colorCycleSpeed;
  float noiseTime = timeSeconds * params.noiseSpeed;

  noStroke();
  rectMode(CENTER);

  for (int y = 0; y < height; y += params.cellSize) {
    for (int x = 0; x < width; x += params.cellSize) {
      float nx = x * params.noiseScale;
      float ny = y * params.noiseScale;
      float n = noise(nx, ny, noiseTime);

      color texColour = samplePalette(n, paletteTime);
      float alpha = params.fillAlpha;
      fill(hue(texColour), saturation(texColour), brightness(texColour), alpha);

      pushMatrix();
      translate(x + params.cellSize * 0.5, y + params.cellSize * 0.5);
      float angle = (n - 0.5) * PI * params.textureDepth;
      rotate(angle);

      float widthScale = 0.65 + n * params.textureDepth;
      float heightScale = 0.75 + sin(paletteTime + n * TWO_PI) * 0.25;
      float rectWidth = params.cellSize * widthScale;
      float rectHeight = params.cellSize * heightScale;
      rect(0, 0, rectWidth, rectHeight, params.cornerRadius);

      if (n > params.accentThreshold) {
        drawAccentLayer(n, paletteTime);
      }
      popMatrix();
    }
  }

  drawNoiseOverlay(timeSeconds);
}

void drawNoiseOverlay(float timeSeconds) {
  pushStyle();
  blendMode(ADD);
  noFill();
  float overlayAlpha = params.overlayAlpha;
  for (int i = 0; i < params.overlayBands; i++) {
    float t = (float) i / max(1, params.overlayBands - 1);
    float noiseOffset = noise(timeSeconds * params.overlaySpeed, t * 10.0);
    color bandColour = samplePalette((t + noiseOffset) % 1.0, timeSeconds);
    stroke(hue(bandColour), saturation(bandColour), brightness(bandColour), overlayAlpha);
    strokeWeight(1.0 + t * params.overlayStrokeScale);

    float yBase = height * (t + 0.05f * sin(TWO_PI * (t + timeSeconds * params.overlaySpeed)));
    beginShape();
    for (int x = 0; x < width; x += params.overlayStep) {
      float distort = noise(x * params.overlayNoiseScale, t * params.overlayNoiseScale, timeSeconds);
      float y = yBase + map(distort, 0, 1, -params.overlayAmplitude, params.overlayAmplitude);
      curveVertex(x, y);
    }
    endShape();
  }
  popStyle();
}

void drawAccentLayer(float noiseValue, float paletteTime) {
  pushStyle();
  noFill();
  strokeWeight(params.accentStrokeWeight);

  for (int layer = 0; layer < params.accentLayers; layer++) {
    float blend = (float) layer / max(1, params.accentLayers - 1);
    float samplePoint = (noiseValue + blend * params.accentJitter) % 1.0;
    color c = samplePalette(samplePoint, paletteTime + blend);
    float alpha = params.accentAlpha * (0.6 + 0.4 * sin(paletteTime + blend * TWO_PI));
    stroke(hue(c), saturation(c), brightness(c), alpha);

    float radius = params.cellSize * (1.0 + blend * params.accentScale);
    float offset = (noiseValue - 0.5) * params.accentDrift * params.cellSize;
    ellipse(offset, offset, radius, radius * (0.7 + 0.3 * cos(paletteTime + blend * PI)));
  }
  popStyle();
}

color samplePalette(float mix, float timePhase) {
  if (params.palette.size() == 0) {
    return color(0, 0, 100, 100);
  }

  float wrappedMix = (mix % 1.0 + 1.0) % 1.0;
  int paletteSize = params.palette.size();
  float scaled = wrappedMix * paletteSize;
  int indexA = floor(scaled) % paletteSize;
  int indexB = (indexA + 1) % paletteSize;
  float localBlend = scaled - floor(scaled);

  float[] colourA = params.palette.get(indexA);
  float[] colourB = params.palette.get(indexB);

  float hueValue = lerpAngle(colourA[0], colourB[0], localBlend);
  hueValue = (hueValue + params.paletteSpread * sin(timePhase + wrappedMix * TWO_PI)) % 360;
  if (hueValue < 0) {
    hueValue += 360;
  }

  float saturationValue = lerp(colourA[1], colourB[1], localBlend);
  float brightnessValue = lerp(colourA[2], colourB[2], localBlend);

  saturationValue = constrain(
    saturationValue + params.saturationJitter * sin(TWO_PI * wrappedMix + timePhase),
    0,
    100
  );
  brightnessValue = constrain(
    brightnessValue + params.brightnessJitter * cos(TWO_PI * wrappedMix + timePhase),
    0,
    100
  );

  return color(hueValue, saturationValue, brightnessValue, 100);
}

float lerpAngle(float start, float end, float amt) {
  float delta = ((end - start + 540) % 360) - 180;
  return (start + delta * amt + 360) % 360;
}

void keyPressed() {
  if (key == 'r' || key == 'R') {
    println("Reloading parameters from " + PARAM_FILE);
    loadParamsJSON();
  }
}

int readInt(JSONObject json, String key, int fallback) {
  return json.hasKey(key) ? json.getInt(key) : fallback;
}

float readFloat(JSONObject json, String key, float fallback) {
  return json.hasKey(key) ? json.getFloat(key) : fallback;
}

void loadParamsJSON() {
  loadParamsJSON(PARAM_FILE, sketchReady);
}

void loadParamsJSON(String path) {
  loadParamsJSON(path, sketchReady);
}

void loadParamsJSON(String path, boolean applyRuntime) {
  SketchParams newParams = new SketchParams();
  JSONObject json = null;
  try {
    json = loadJSONObject(path);
    println("Parameters loaded from " + path);
  } catch (Exception e) {
    println("Could not load parameters from " + path + ". Using defaults.");
  }

  if (json != null) {
    newParams.loadFrom(json);
  }

  params = newParams;

  if (applyRuntime) {
    applyRuntimeParams();
  }
}

void applyRuntimeParams() {
  if (surface != null) {
    surface.setSize(params.canvasWidth, params.canvasHeight);
  }
  smooth(params.smoothingSamples);
  frameRate(params.targetFrameRate);
  noiseDetail(params.noiseOctaves, params.noiseFalloff);
}

class SketchParams {
  int canvasWidth = 960;
  int canvasHeight = 960;
  int cellSize = 24;

  float noiseScale = 0.006;
  float noiseSpeed = 0.2;
  int noiseOctaves = 4;
  float noiseFalloff = 0.5;

  float colorCycleSpeed = 0.4;
  float paletteSpread = 80;
  float saturationJitter = 20;
  float brightnessJitter = 25;

  float textureDepth = 0.7;
  float fillAlpha = 60;
  float cornerRadius = 4.0;

  float accentThreshold = 0.72;
  int accentLayers = 3;
  float accentScale = 2.2;
  float accentStrokeWeight = 1.6;
  float accentAlpha = 45;
  float accentJitter = 0.35;
  float accentDrift = 0.45;

  int overlayBands = 5;
  float overlayNoiseScale = 0.004;
  float overlayAmplitude = 40;
  float overlayAlpha = 28;
  float overlaySpeed = 0.35;
  int overlayStep = 16;
  float overlayStrokeScale = 3.0;

  float backgroundHue = 210;
  float backgroundSaturation = 18;
  float backgroundBrightness = 8;
  float backgroundAlpha = 100;

  int targetFrameRate = 60;
  int smoothingSamples = 4;

  ArrayList<float[]> palette = new ArrayList<float[]>();

  SketchParams() {
    palette.add(new float[] {210, 65, 95});
    palette.add(new float[] {320, 70, 95});
    palette.add(new float[] {35, 85, 95});
    palette.add(new float[] {160, 55, 90});
  }

  void loadFrom(JSONObject json) {
    canvasWidth = readInt(json, "canvasWidth", canvasWidth);
    canvasHeight = readInt(json, "canvasHeight", canvasHeight);
    cellSize = readInt(json, "cellSize", cellSize);

    noiseScale = readFloat(json, "noiseScale", noiseScale);
    noiseSpeed = readFloat(json, "noiseSpeed", noiseSpeed);
    noiseOctaves = readInt(json, "noiseOctaves", noiseOctaves);
    noiseFalloff = readFloat(json, "noiseFalloff", noiseFalloff);

    colorCycleSpeed = readFloat(json, "colorCycleSpeed", colorCycleSpeed);
    paletteSpread = readFloat(json, "paletteSpread", paletteSpread);
    saturationJitter = readFloat(json, "saturationJitter", saturationJitter);
    brightnessJitter = readFloat(json, "brightnessJitter", brightnessJitter);

    textureDepth = readFloat(json, "textureDepth", textureDepth);
    fillAlpha = readFloat(json, "fillAlpha", fillAlpha);
    cornerRadius = readFloat(json, "cornerRadius", cornerRadius);

    accentThreshold = readFloat(json, "accentThreshold", accentThreshold);
    accentLayers = readInt(json, "accentLayers", accentLayers);
    accentScale = readFloat(json, "accentScale", accentScale);
    accentStrokeWeight = readFloat(json, "accentStrokeWeight", accentStrokeWeight);
    accentAlpha = readFloat(json, "accentAlpha", accentAlpha);
    accentJitter = readFloat(json, "accentJitter", accentJitter);
    accentDrift = readFloat(json, "accentDrift", accentDrift);

    overlayBands = readInt(json, "overlayBands", overlayBands);
    overlayNoiseScale = readFloat(json, "overlayNoiseScale", overlayNoiseScale);
    overlayAmplitude = readFloat(json, "overlayAmplitude", overlayAmplitude);
    overlayAlpha = readFloat(json, "overlayAlpha", overlayAlpha);
    overlaySpeed = readFloat(json, "overlaySpeed", overlaySpeed);
    overlayStep = readInt(json, "overlayStep", overlayStep);
    overlayStrokeScale = readFloat(json, "overlayStrokeScale", overlayStrokeScale);

    backgroundHue = readFloat(json, "backgroundHue", backgroundHue);
    backgroundSaturation = readFloat(json, "backgroundSaturation", backgroundSaturation);
    backgroundBrightness = readFloat(json, "backgroundBrightness", backgroundBrightness);
    backgroundAlpha = readFloat(json, "backgroundAlpha", backgroundAlpha);

    targetFrameRate = readInt(json, "targetFrameRate", targetFrameRate);
    smoothingSamples = readInt(json, "smoothingSamples", smoothingSamples);

    if (json.hasKey("palette")) {
      JSONArray arr = json.getJSONArray("palette");
      if (arr != null && arr.size() > 0) {
        palette = new ArrayList<float[]>();
        for (int i = 0; i < arr.size(); i++) {
          JSONObject entry = arr.getJSONObject(i);
          if (entry == null) {
            continue;
          }
          float h = entry.hasKey("h") ? entry.getFloat("h") : Float.NaN;
          float s = entry.hasKey("s") ? entry.getFloat("s") : Float.NaN;
          float b = entry.hasKey("b") ? entry.getFloat("b") : Float.NaN;
          if (!Float.isNaN(h) && !Float.isNaN(s) && !Float.isNaN(b)) {
            palette.add(new float[] {h, s, b});
          }
        }
        if (palette.size() == 1) {
          float[] colour = palette.get(0);
          palette.add(new float[] {(colour[0] + 120) % 360, colour[1], colour[2]});
        }
      }
    }
  }
}
