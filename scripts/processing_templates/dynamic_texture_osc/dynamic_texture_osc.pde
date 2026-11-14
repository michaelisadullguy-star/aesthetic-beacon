// Aesthetic Beacon — live display
//
// A multi-layered, colour-rich texture driven by animated Perlin noise.  Its
// 50 parameters are set by the Python feedback loop:
//
//   1. over OSC (UDP port 12000) — no extra library needed, and
//   2. from the JSON file the loop keeps up to date, which this sketch watches
//      as a fallback in case UDP is blocked.
//
// When no update arrives the sketch simply keeps animating the last
// parameter set, which is exactly the intended behaviour while the creator is
// not in a "creatable" state.  New parameter sets are eased in over
// TRANSITION_SECONDS so the picture never jumps.
//
// Keys:  O  status overlay      R  reload data/dynamic_texture_params.json
//        S  save frames as PNG  (frames/frame_######.png) on/off
//
// OSC messages understood (prefix /aesthetic_beacon):
//   /params  <sequence:int> <name:string> <value> <name:string> <value> ...
//   /state   <creatable:int> <confidence:float> <label:string>
//   /<name>  <value>                      a single parameter, handy for testing
//
// Parameter names, ranges and defaults come from data/parameter_schema.json,
// which is generated from src/parameters.py (python -m src.parameters).

import java.io.File;
import java.util.HashMap;

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

final int OSC_PORT = 12000;
final String OSC_PREFIX = "/aesthetic_beacon";

// P2D uses the graphics card. If the window stays black or the sketch fails
// to start, change this to JAVA2D.
final String RENDERER = P2D;

// Seconds over which a new parameter set is blended in.
final float TRANSITION_SECONDS = 1.5;

final String PARAM_FILE = "data/dynamic_texture_params.json";
final String SCHEMA_FILE = "data/parameter_schema.json";

// File written by the Python loop, relative to this sketch's folder.
// Use an absolute path if the sketch is moved out of the repository.
final String LIVE_PARAM_FILE = "../../../data/step6_realtime/outputs/current_parameters.json";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

int canvasWidth = 1080;
int canvasHeight = 640;
int targetFrameRate = 60;
int smoothingSamples = 4;

ArrayList<ParamSpec> specs = new ArrayList<ParamSpec>();
HashMap<String, ParamSpec> specByName = new HashMap<String, ParamSpec>();
SketchParams params = new SketchParams();

ArrayList<CellData> cellGrid = new ArrayList<CellData>();
ArrayList<OverlayBandData> overlayBandData = new ArrayList<OverlayBandData>();
int builtCellSize = -1;
int builtWidth = -1;
int builtHeight = -1;
int builtOverlayBands = -1;

OscReceiver receiver;
File liveFile;
long liveFileModified = 0;
int lastLivePoll = -10000;
long lastSequence = -1;

boolean showStatus = false;
boolean exportFrames = false;
boolean isCreatableState = false;
float creatableConfidence = 0;
String stateLabel = "waiting";
String lastTransport = "none";
int lastUpdateMillis = -1;
int updatesReceived = 0;
int lastFrameMillis = 0;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void settings() {
  // Canvas settings are fixed for the session; they are read once, here.
  try {
    JSONObject json = loadJSONObject(PARAM_FILE);
    canvasWidth = constrain(readInt(json, "canvasWidth", canvasWidth), 64, 4096);
    canvasHeight = constrain(readInt(json, "canvasHeight", canvasHeight), 64, 4096);
    targetFrameRate = constrain(readInt(json, "targetFrameRate", targetFrameRate), 1, 240);
    smoothingSamples = readInt(json, "smoothingSamples", smoothingSamples);
  } catch (Exception e) {
    println("Could not read canvas settings from " + PARAM_FILE + "; using 1080x640 at 60 fps.");
  }
  size(canvasWidth, canvasHeight, RENDERER);
  if (smoothingSamples >= 2) {
    smooth(smoothingSamples >= 8 ? 8 : (smoothingSamples >= 4 ? 4 : 2));
  } else {
    noSmooth();
  }
}

void setup() {
  colorMode(HSB, 360, 100, 100, 100);
  frameRate(targetFrameRate);

  if (!loadSchema()) {
    println("ERROR: " + SCHEMA_FILE + " is missing or unreadable. Run 'python -m src.parameters' in the repository.");
    exit();
    return;
  }
  loadParamFile(PARAM_FILE, true);
  params.sync();

  receiver = new OscReceiver(OSC_PORT);
  if (receiver.isListening()) {
    println("Listening for OSC on UDP port " + OSC_PORT + " (prefix " + OSC_PREFIX + ")");
  } else {
    println("Could not open UDP port " + OSC_PORT + " (" + receiver.getError() + ").");
    println("Is another copy of the sketch running? Falling back to the parameter file only.");
  }

  liveFile = new File(sketchPath(LIVE_PARAM_FILE));
  println("Watching " + liveFile.getAbsolutePath());
  if (liveFile.exists()) {
    // Ignore a file left over from an earlier session until it changes.
    liveFileModified = liveFile.lastModified();
  }
  lastFrameMillis = millis();
}

void draw() {
  int now = millis();
  float dt = constrain((now - lastFrameMillis) * 0.001, 0, 0.25);
  lastFrameMillis = now;

  drainOscMessages();
  pollLiveFile(now);
  easeParameters(dt);
  params.sync();
  rebuildIfNeeded();

  background(
    params.backgroundHue,
    params.backgroundSaturation,
    params.backgroundBrightness,
    params.backgroundAlpha
  );

  float timeSeconds = now * 0.001;
  float paletteTime = timeSeconds * params.colorCycleSpeed;
  float noiseTime = timeSeconds * params.noiseSpeed;
  noiseDetail(params.noiseOctaves, params.noiseFalloff);

  noStroke();
  rectMode(CENTER);

  for (CellData cell : cellGrid) {
    float noiseValue = noise(cell.gridX * params.noiseScale, cell.gridY * params.noiseScale, noiseTime);
    float paletteSample = fract(noiseValue + cell.paletteSeed * params.palettePhaseJitter + sin(paletteTime + cell.paletteSeed * TWO_PI) * 0.1);
    color texColour = samplePalette(paletteSample, paletteTime + cell.paletteSeed * params.palettePhaseJitter * TWO_PI);

    float alphaOsc = 0.75 + 0.25 * sin(paletteTime * 1.1 + cell.paletteSeed * TWO_PI);
    fill(hue(texColour), saturation(texColour), brightness(texColour), params.fillAlpha * alphaOsc);

    float offsetAngle = paletteTime * 0.8 + cell.offsetSeed * TWO_PI;
    float offsetRadius = params.cellOffsetJitter * params.cellSize * sin(paletteTime + cell.offsetSeed * TWO_PI);
    float px = cell.centerX + cos(offsetAngle) * offsetRadius;
    float py = cell.centerY + sin(offsetAngle) * offsetRadius;

    pushMatrix();
    translate(px, py);
    float baseAngle = (noiseValue - 0.5) * PI * params.textureDepth;
    rotate(baseAngle + cell.rotationSeed * params.cellRotationJitter);

    float widthScale = 0.7 + noiseValue * params.textureDepth + cell.widthJitter * params.cellScaleJitter;
    float heightScale = 0.7 + cos(paletteTime + noiseValue * PI) * 0.3 + cell.heightJitter * params.cellScaleJitter;
    float rectWidth = params.cellSize * max(0.25, widthScale);
    float rectHeight = params.cellSize * max(0.25, heightScale);
    float corner = max(0, params.cornerRadius + cell.widthJitter * params.cornerRadiusJitter);
    corner = min(corner, min(rectWidth, rectHeight) * 0.5);
    rect(0, 0, rectWidth, rectHeight, corner);

    if (params.accentLayers > 0 && noiseValue > params.accentThreshold) {
      drawAccentLayer(noiseValue, paletteTime, cell);
    }
    popMatrix();
  }

  drawNoiseOverlay(timeSeconds);

  if (exportFrames) {
    saveFrame("frames/frame_######.png");
  }
  if (showStatus) {
    drawStatus(now);
  }
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

void drawNoiseOverlay(float timeSeconds) {
  if (overlayBandData.size() == 0) {
    return;
  }
  int step = max(1, params.overlayStep);

  pushStyle();
  blendMode(ADD);
  noFill();
  for (OverlayBandData band : overlayBandData) {
    float timePhase = timeSeconds * params.overlaySpeed + band.noiseShift;
    float noiseOffset = noise(timePhase, band.offsetSeed * 11.0);
    float paletteSample = fract(band.position + noiseOffset + band.offsetSeed);
    color bandColour = samplePalette(paletteSample, timeSeconds + band.offsetSeed * params.overlayAlphaJitter * TWO_PI);

    float alphaOsc = 0.65 + 0.35 * sin(timeSeconds * 0.7 + band.offsetSeed * TWO_PI);
    stroke(hue(bandColour), saturation(bandColour), brightness(bandColour), params.overlayAlpha * alphaOsc);
    strokeWeight(1.0 + band.strokeSeed * params.overlayStrokeScale);

    float yBase = height * (band.position + 0.05f * sin(TWO_PI * (band.position + timePhase + band.offsetSeed)));
    beginShape();
    for (int x = -step; x <= width + step; x += step) {
      float distort = noise(x * params.overlayNoiseScale, band.position * 7.0, timePhase + band.noiseShift);
      float y = yBase + map(distort, 0, 1, -params.overlayAmplitude, params.overlayAmplitude);
      curveVertex(x, y);
    }
    endShape();
  }
  blendMode(BLEND);
  popStyle();
}

void drawAccentLayer(float noiseValue, float paletteTime, CellData cell) {
  pushStyle();
  noFill();
  strokeWeight(params.accentStrokeWeight);

  for (int layer = 0; layer < params.accentLayers; layer++) {
    float blend = (float) layer / max(1, params.accentLayers - 1);
    float samplePoint = fract(noiseValue + blend * params.accentJitter + cell.accentSeed);
    color c = samplePalette(samplePoint, paletteTime + blend + cell.accentSeed * params.accentPhaseJitter * TWO_PI);
    float phase = paletteTime * 0.9 + blend * TWO_PI + cell.accentSeed * params.accentPhaseJitter * TWO_PI;
    float alpha = params.accentAlpha * (0.55 + 0.45 * sin(phase));
    stroke(hue(c), saturation(c), brightness(c), alpha);

    float radiusBase = params.cellSize * (1.0 + blend * params.accentScale);
    float radius = radiusBase * (0.9 + 0.2 * sin(phase + cell.rotationSeed));
    float offset = (noiseValue - 0.5) * params.accentDrift * params.cellSize + cell.rotationSeed * params.accentRadiusJitter * params.cellSize * 0.5;
    float skew = 0.7 + 0.3 * cos(phase + cell.offsetSeed * TWO_PI);
    ellipse(offset, offset * skew, radius, radius * skew);
  }
  popStyle();
}

void drawStatus(int now) {
  pushStyle();
  rectMode(CORNER);
  noStroke();
  fill(0, 0, 0, 70);
  rect(10, 10, 330, 118, 6);

  fill(0, 0, 100, 100);
  textAlign(LEFT, TOP);
  textSize(12);
  String port = (receiver != null && receiver.isListening()) ? ("UDP " + OSC_PORT) : "UDP unavailable";
  String age = lastUpdateMillis < 0 ? "never" : nf((now - lastUpdateMillis) / 1000.0, 1, 1) + " s ago";
  text("OSC: " + port, 20, 18);
  text("State: " + stateLabel.toUpperCase().replace('_', ' '), 20, 36);
  text("Confidence: " + nf(creatableConfidence, 1, 2), 20, 54);
  text("Last parameter update: " + age + " via " + lastTransport, 20, 72);
  text("Updates received: " + updatesReceived, 20, 90);
  text(nf(frameRate, 1, 0) + " fps" + (exportFrames ? "   saving frames" : ""), 20, 108);

  fill(isCreatableState ? color(120, 80, 90) : color(0, 0, 55));
  ellipse(320, 28, 14, 14);
  popStyle();
}

color samplePalette(float mix, float timePhase) {
  int paletteSize = params.palette.length;
  float wrappedMix = (mix % 1.0 + 1.0) % 1.0;
  float scaled = wrappedMix * paletteSize;
  int indexA = floor(scaled) % paletteSize;
  int indexB = (indexA + 1) % paletteSize;
  float localBlend = scaled - floor(scaled);

  float[] colourA = params.palette[indexA];
  float[] colourB = params.palette[indexB];

  float hueValue = lerpAngle(colourA[0], colourB[0], localBlend);
  hueValue = (hueValue + params.paletteSpread * sin(timePhase + wrappedMix * TWO_PI)) % 360;
  if (hueValue < 0) {
    hueValue += 360;
  }

  float saturationValue = constrain(
    lerp(colourA[1], colourB[1], localBlend) + params.saturationJitter * sin(TWO_PI * wrappedMix + timePhase),
    0,
    100
  );
  float brightnessValue = constrain(
    lerp(colourA[2], colourB[2], localBlend) + params.brightnessJitter * cos(TWO_PI * wrappedMix + timePhase),
    0,
    100
  );
  return color(hueValue, saturationValue, brightnessValue, 100);
}

float lerpAngle(float start, float end, float amt) {
  float delta = ((end - start + 540) % 360) - 180;
  return (start + delta * amt + 360) % 360;
}

// ---------------------------------------------------------------------------
// Receiving parameters
// ---------------------------------------------------------------------------

void keyPressed() {
  if (key == 'r' || key == 'R') {
    println("Reloading parameters from " + PARAM_FILE);
    loadParamFile(PARAM_FILE, false);
  } else if (key == 'o' || key == 'O') {
    showStatus = !showStatus;
  } else if (key == 's' || key == 'S') {
    exportFrames = !exportFrames;
    println("Saving frames: " + (exportFrames ? "ON (frames/ in the sketch folder)" : "OFF"));
  }
}

// Called once per frame; applies everything the network thread has queued.
void drainOscMessages() {
  if (receiver == null) {
    return;
  }
  OscReceiver.Message message;
  while ((message = receiver.poll()) != null) {
    handleMessage(message.address, message.args);
  }
}

void handleMessage(String address, Object[] args) {
  if (!address.startsWith(OSC_PREFIX + "/")) {
    return;
  }
  String name = address.substring(OSC_PREFIX.length() + 1);

  if (name.equals("params")) {
    int index = 0;
    if (args.length % 2 == 1 && args[0] instanceof Number) {
      lastSequence = ((Number) args[0]).longValue();
      index = 1;
    }
    int applied = 0;
    for (; index + 1 < args.length; index += 2) {
      if (args[index] instanceof String && args[index + 1] instanceof Number) {
        if (setTarget((String) args[index], ((Number) args[index + 1]).floatValue())) {
          applied++;
        }
      }
    }
    if (applied > 0) {
      markUpdate("OSC");
    }
    return;
  }

  if (name.equals("state")) {
    if (args.length >= 1 && args[0] instanceof Number) {
      isCreatableState = ((Number) args[0]).intValue() > 0;
    }
    if (args.length >= 2 && args[1] instanceof Number) {
      creatableConfidence = ((Number) args[1]).floatValue();
    }
    if (args.length >= 3 && args[2] instanceof String) {
      stateLabel = (String) args[2];
    } else {
      stateLabel = isCreatableState ? "creatable" : "idle";
    }
    return;
  }

  if (args.length >= 1 && args[0] instanceof Number) {
    if (setTarget(name, ((Number) args[0]).floatValue())) {
      markUpdate("OSC");
    }
  }
}

void markUpdate(String transport) {
  lastTransport = transport;
  lastUpdateMillis = millis();
  updatesReceived++;
}

// Checks twice a second whether the Python loop has rewritten its parameter file.
void pollLiveFile(int now) {
  if (liveFile == null || now - lastLivePoll < 500) {
    return;
  }
  lastLivePoll = now;
  long modified = liveFile.lastModified(); // 0 when the file does not exist
  if (modified == 0 || modified == liveFileModified) {
    return;
  }
  liveFileModified = modified;
  try {
    JSONObject json = loadJSONObject(liveFile.getAbsolutePath());
    long sequence = json.hasKey("sequence") ? json.getLong("sequence") : -2;
    if (json.hasKey("state")) {
      JSONObject state = json.getJSONObject("state");
      // Only trust the file for the state when OSC is silent.
      if (lastUpdateMillis < 0 || now - lastUpdateMillis > 8000) {
        isCreatableState = state.getBoolean("creatable", false);
        creatableConfidence = state.getFloat("confidence", 0);
        stateLabel = state.getString("label", isCreatableState ? "creatable" : "idle");
      }
    }
    if (sequence == lastSequence) {
      return; // already applied via OSC
    }
    if (applyJson(json, false) > 0) {
      lastSequence = sequence;
      markUpdate("file");
    }
  } catch (Exception e) {
    // The file is replaced atomically, so this is rare; try again on the next change.
  }
}

// ---------------------------------------------------------------------------
// Parameter store
// ---------------------------------------------------------------------------

boolean loadSchema() {
  try {
    JSONArray array = loadJSONArray(SCHEMA_FILE);
    for (int i = 0; i < array.size(); i++) {
      JSONObject entry = array.getJSONObject(i);
      ParamSpec spec = new ParamSpec();
      spec.name = entry.getString("name");
      spec.minValue = entry.getFloat("min");
      spec.maxValue = entry.getFloat("max");
      spec.integer = entry.getBoolean("integer", false);
      spec.circular = entry.getBoolean("circular", false);
      spec.target = spec.clean(entry.getFloat("default"));
      spec.current = spec.target;
      specs.add(spec);
      specByName.put(spec.name, spec);
    }
    return specs.size() > 0;
  } catch (Exception e) {
    return false;
  }
}

// Sets the value a parameter should move towards. Unknown names and
// non-finite values are ignored; everything else is clamped to its range.
boolean setTarget(String name, float value) {
  ParamSpec spec = specByName.get(name);
  if (spec == null || Float.isNaN(value) || Float.isInfinite(value)) {
    return false;
  }
  spec.target = spec.clean(value);
  return true;
}

int applyJson(JSONObject json, boolean immediately) {
  int applied = 0;
  for (ParamSpec spec : specs) {
    if (json.hasKey(spec.name)) {
      try {
        if (setTarget(spec.name, json.getFloat(spec.name))) {
          applied++;
        }
      } catch (Exception e) {
        // not a number: skip
      }
    }
  }
  if (json.hasKey("palette")) {
    try {
      JSONArray palette = json.getJSONArray("palette");
      String[] components = {"h", "s", "b"};
      for (int i = 0; i < palette.size() && i < 4; i++) {
        JSONObject entry = palette.getJSONObject(i);
        for (String component : components) {
          if (entry.hasKey(component) && setTarget("palette_" + (i + 1) + "_" + component, entry.getFloat(component))) {
            applied++;
          }
        }
      }
    } catch (Exception e) {
      // malformed palette: keep the current one
    }
  }
  if (immediately) {
    for (ParamSpec spec : specs) {
      spec.current = spec.target;
    }
  }
  return applied;
}

void loadParamFile(String path, boolean immediately) {
  try {
    JSONObject json = loadJSONObject(path);
    int applied = applyJson(json, immediately);
    println("Loaded " + applied + " parameters from " + path);
  } catch (Exception e) {
    println("Could not load parameters from " + path + ". Keeping the current values.");
  }
}

void easeParameters(float dt) {
  float amount = TRANSITION_SECONDS <= 0 ? 1.0 : 1.0 - exp(-dt / (TRANSITION_SECONDS / 3.0));
  for (ParamSpec spec : specs) {
    if (spec.integer) {
      spec.current = spec.target;
    } else if (spec.circular) {
      float delta = ((spec.target - spec.current + 1.5 * spec.maxValue) % spec.maxValue) - 0.5 * spec.maxValue;
      spec.current = (spec.current + delta * amount + spec.maxValue) % spec.maxValue;
    } else {
      spec.current += (spec.target - spec.current) * amount;
    }
  }
}

float value(String name, float fallback) {
  ParamSpec spec = specByName.get(name);
  return spec == null ? fallback : spec.current;
}

void rebuildIfNeeded() {
  if (params.cellSize != builtCellSize || width != builtWidth || height != builtHeight) {
    buildCellGrid();
  }
  if (params.overlayBands != builtOverlayBands) {
    buildOverlayBands();
  }
}

void buildCellGrid() {
  int cellSize = max(4, params.cellSize); // a step of zero would never finish
  ArrayList<CellData> grid = new ArrayList<CellData>();
  for (int y = 0; y < height; y += cellSize) {
    for (int x = 0; x < width; x += cellSize) {
      CellData cell = new CellData();
      cell.gridX = x;
      cell.gridY = y;
      cell.centerX = x + cellSize * 0.5;
      cell.centerY = y + cellSize * 0.5;
      cell.widthJitter = signedRandom(pseudoRandom(x, y, 11.0));
      cell.heightJitter = signedRandom(pseudoRandom(x, y, 19.0));
      cell.rotationSeed = signedRandom(pseudoRandom(x, y, 23.0));
      cell.offsetSeed = pseudoRandom(x, y, 31.0);
      cell.paletteSeed = pseudoRandom(x, y, 47.0);
      cell.accentSeed = pseudoRandom(x, y, 59.0);
      grid.add(cell);
    }
  }
  cellGrid = grid;
  builtCellSize = params.cellSize;
  builtWidth = width;
  builtHeight = height;
}

void buildOverlayBands() {
  ArrayList<OverlayBandData> bands = new ArrayList<OverlayBandData>();
  int count = constrain(params.overlayBands, 0, 64);
  for (int i = 0; i < count; i++) {
    OverlayBandData band = new OverlayBandData();
    band.position = count == 1 ? 0.5 : (float) i / (float) (count - 1);
    band.offsetSeed = pseudoRandom(i, count, 71.0);
    band.strokeSeed = 0.5 + 0.5 * signedRandom(pseudoRandom(i, count, 83.0));
    band.noiseShift = pseudoRandom(i, count, 97.0) * 8.0;
    bands.add(band);
  }
  overlayBandData = bands;
  builtOverlayBands = params.overlayBands;
}

// ---------------------------------------------------------------------------
// Helpers and data classes
// ---------------------------------------------------------------------------

int readInt(JSONObject json, String key, int fallback) {
  try {
    return json.hasKey(key) ? json.getInt(key) : fallback;
  } catch (Exception e) {
    return fallback;
  }
}

float fract(float v) {
  return v - floor(v);
}

float signedRandom(float r) {
  return r * 2.0 - 1.0;
}

float pseudoRandom(float x, float y, double salt) {
  float v = sin((float) ((x + salt) * 12.9898 + (y - salt) * 78.233));
  return fract(v * 43758.5453f);
}

class ParamSpec {
  String name;
  float minValue;
  float maxValue;
  boolean integer;
  boolean circular;
  float current;
  float target;

  float clean(float v) {
    if (circular) {
      v = ((v % maxValue) + maxValue) % maxValue;
    } else {
      v = constrain(v, minValue, maxValue);
    }
    return integer ? round(v) : v;
  }
}

class CellData {
  float gridX;
  float gridY;
  float centerX;
  float centerY;
  float widthJitter;
  float heightJitter;
  float rotationSeed;
  float offsetSeed;
  float paletteSeed;
  float accentSeed;
}

class OverlayBandData {
  float position;
  float offsetSeed;
  float strokeSeed;
  float noiseShift;
}

// The current (eased) parameter values as plain fields for the drawing code.
// The fallbacks only matter if a name is missing from the schema file.
class SketchParams {
  int cellSize = 24;
  float cellScaleJitter, cellRotationJitter, cellOffsetJitter, cornerRadiusJitter, palettePhaseJitter;
  float noiseScale = 0.006, noiseSpeed, noiseFalloff = 0.5;
  int noiseOctaves = 4;
  float colorCycleSpeed, paletteSpread, saturationJitter, brightnessJitter;
  float textureDepth, fillAlpha = 60, cornerRadius;
  float accentThreshold = 0.72, accentScale, accentStrokeWeight = 1, accentAlpha, accentJitter, accentDrift, accentPhaseJitter, accentRadiusJitter;
  int accentLayers;
  int overlayBands, overlayStep = 16;
  float overlayNoiseScale, overlayAmplitude, overlayAlpha, overlaySpeed, overlayStrokeScale, overlayAlphaJitter;
  float backgroundHue, backgroundSaturation, backgroundBrightness, backgroundAlpha = 100;
  float[][] palette = {{210, 65, 95}, {310, 70, 95}, {35, 85, 96}, {160, 55, 90}};

  void sync() {
    cellSize = max(4, round(value("cellSize", 24)));
    cellScaleJitter = value("cellScaleJitter", 0.38);
    cellRotationJitter = value("cellRotationJitter", 0.55);
    cellOffsetJitter = value("cellOffsetJitter", 0.3);
    cornerRadiusJitter = value("cornerRadiusJitter", 2.8);
    palettePhaseJitter = value("palettePhaseJitter", 0.28);

    noiseScale = value("noiseScale", 0.006);
    noiseSpeed = value("noiseSpeed", 0.18);
    noiseOctaves = constrain(round(value("noiseOctaves", 5)), 1, 8);
    noiseFalloff = constrain(value("noiseFalloff", 0.52), 0.05, 0.95);

    colorCycleSpeed = value("colorCycleSpeed", 0.45);
    paletteSpread = value("paletteSpread", 85);
    saturationJitter = value("saturationJitter", 18);
    brightnessJitter = value("brightnessJitter", 26);

    textureDepth = value("textureDepth", 0.7);
    fillAlpha = value("fillAlpha", 60);
    cornerRadius = value("cornerRadius", 5);

    accentThreshold = value("accentThreshold", 0.72);
    accentLayers = constrain(round(value("accentLayers", 3)), 0, 12);
    accentScale = value("accentScale", 2.2);
    accentStrokeWeight = max(0.1, value("accentStrokeWeight", 1.6));
    accentAlpha = value("accentAlpha", 45);
    accentJitter = value("accentJitter", 0.35);
    accentDrift = value("accentDrift", 0.5);
    accentPhaseJitter = value("accentPhaseJitter", 0.4);
    accentRadiusJitter = value("accentRadiusJitter", 0.4);

    overlayBands = constrain(round(value("overlayBands", 6)), 0, 64);
    overlayNoiseScale = value("overlayNoiseScale", 0.0035);
    overlayAmplitude = value("overlayAmplitude", 42);
    overlayAlpha = value("overlayAlpha", 32);
    overlaySpeed = value("overlaySpeed", 0.4);
    overlayStep = max(1, round(value("overlayStep", 14)));
    overlayStrokeScale = value("overlayStrokeScale", 3.4);
    overlayAlphaJitter = value("overlayAlphaJitter", 0.4);

    backgroundHue = value("backgroundHue", 210);
    backgroundSaturation = value("backgroundSaturation", 18);
    backgroundBrightness = value("backgroundBrightness", 8);
    backgroundAlpha = value("backgroundAlpha", 100);

    for (int i = 0; i < 4; i++) {
      palette[i][0] = value("palette_" + (i + 1) + "_h", palette[i][0]);
      palette[i][1] = value("palette_" + (i + 1) + "_s", palette[i][1]);
      palette[i][2] = value("palette_" + (i + 1) + "_b", palette[i][2]);
    }
  }
}
