import { StatusBar } from 'expo-status-bar';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  StyleSheet, Text, View, ScrollView, useWindowDimensions,
  TouchableOpacity, ActivityIndicator, TextInput, Platform, Modal
} from 'react-native';
import { Picker } from '@react-native-picker/picker';
import FeatureDetails from './FeatureDetails';

const DEFAULT_TEXT = "Did you know that pineapples were a symbol of hospitality in colonial America? This exotic fruit, once a rare delicacy, was often displayed at gatherings to impress guests.";
const API_BASE = 'http://localhost:8000';

const TAB_SAE = 'sae';
const TAB_FEATURE_MAP = 'feature-map';
const TAB_FEATURE_TRACE = 'feature-trace';

function sanitizeDecimalInput(value) {
  const cleaned = (value || '').replace(/[^0-9.]/g, '');
  const firstDotIndex = cleaned.indexOf('.');
  if (firstDotIndex === -1) return cleaned;
  return cleaned.slice(0, firstDotIndex + 1) + cleaned.slice(firstDotIndex + 1).replace(/\./g, '');
}

// Hover-aware feature chip component
function FeatureChip({ feature, onPress, compact = false, styles }) {
  const [hovered, setHovered] = useState(false);
  const featureName = feature.name || feature.label || `Feature ${feature.id}`;
  const featureDescription = feature.description || featureName;
  const truncated = featureName.length > 55
    ? featureName.slice(0, 55) + '…'
    : featureName;

  return (
    <View style={{ position: 'relative' }}>
      <TouchableOpacity
        style={[styles.featureCard, compact && styles.featureCardCompact, hovered && styles.featureCardHovered]}
        onPress={onPress}
        {...(Platform.OS === 'web' ? {
          onMouseEnter: () => setHovered(true),
          onMouseLeave: () => setHovered(false),
        } : {})}
      >
        <View style={styles.featureCardTop}>
          <View style={styles.featureIdBadge}>
            <Text style={styles.featureIdText}>#{feature.id}</Text>
          </View>
          <View style={styles.activationPill}>
            <Text style={styles.activationValue}>{feature.activation?.toFixed(3) ?? '—'}</Text>
            <Text style={styles.activationUnit}>ACT</Text>
          </View>
        </View>
        <Text style={styles.featureLabel} numberOfLines={2}>{truncated}</Text>
      </TouchableOpacity>

      {hovered && Platform.OS === 'web' && (
        <View style={styles.tooltipLeft}>
          <Text style={styles.tooltipTitle}>Feature #{feature.id}</Text>
          <Text style={styles.tooltipDesc}>{featureDescription}</Text>
          <Text style={styles.tooltipHint}>Click to open full analysis ↗</Text>
        </View>
      )}
    </View>
  );
}

// Small chip for feature map tab
function SmallFeatureChip({ feat, onPress, styles }) {
  const [hovered, setHovered] = useState(false);
  return (
    <View style={{ position: 'relative' }}>
      <TouchableOpacity
        style={[styles.smallChip, hovered && styles.smallChipHovered]}
        onPress={onPress}
        {...(Platform.OS === 'web' ? {
          onMouseEnter: () => setHovered(true),
          onMouseLeave: () => setHovered(false),
        } : {})}
      >
        <Text style={styles.smallChipId}>#{feat.id}</Text>
        <Text style={styles.smallChipLabel} numberOfLines={1}>
          {feat.label || feat.description || `Feature ${feat.id}`}
        </Text>
      </TouchableOpacity>
      {hovered && Platform.OS === 'web' && (
        <View style={[styles.tooltipLeft, { zIndex: 999 }]}>
          <Text style={styles.tooltipTitle}>Feature #{feat.id}</Text>
          <Text style={styles.tooltipDesc}>{feat.description || feat.label}</Text>
        </View>
      )}
    </View>
  );
}

export default function App() {
  const [themeMode, setThemeMode] = useState('dark');
  const C = themeMode === 'light' ? LIGHT_THEME : DARK_THEME;
  const styles = useMemo(() => createStyles(C), [C]);

  const [activeTab, setActiveTab] = useState(TAB_SAE);
  const [tokens, setTokens] = useState([]);
  const [inputText, setInputText] = useState(DEFAULT_TEXT);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelsLoading, setModelsLoading] = useState(true);

  const [activeTokenIndex, setActiveTokenIndex] = useState(null);
  const [hoveredTokenIndex, setHoveredTokenIndex] = useState(null);
  const [topK, setTopK] = useState(0);
  const [selectedFeature, setSelectedFeature] = useState(null);
  const hoverClearTimerRef = useRef(null);

  // Feature lookup state
  const [lookupFeatureId, setLookupFeatureId] = useState('');
  const [lookupMaxSentences, setLookupMaxSentences] = useState('200');
  const [lookupMaxResults, setLookupMaxResults] = useState('100');
  const [lookupMinActivation, setLookupMinActivation] = useState('0.0');
  const [featureLookupLoading, setFeatureLookupLoading] = useState(false);
  const [featureLookupError, setFeatureLookupError] = useState(null);
  const [featureLookupResults, setFeatureLookupResults] = useState(null);
  const [traceText, setTraceText] = useState('');
  const [traceFeatureId, setTraceFeatureId] = useState('');
  const [traceMinActivation, setTraceMinActivation] = useState('0.0');
  const [traceLoading, setTraceLoading] = useState(false);
  const [traceError, setTraceError] = useState(null);
  const [traceResult, setTraceResult] = useState(null);
  const [traceHoveredIndex, setTraceHoveredIndex] = useState(null);

  const { width } = useWindowDimensions();
  const isLargeScreen = width > 768;

  useEffect(() => {
    const fetchModels = async (retries = 3) => {
      for (let attempt = 1; attempt <= retries; attempt++) {
        try {
          const controller = new AbortController();
          const timeout = setTimeout(() => controller.abort(), 60000);
          const res = await fetch(`${API_BASE}/models?analyzer=sae`, { signal: controller.signal });
          clearTimeout(timeout);
          const data = await res.json();
          const available = (data.models || []).filter(m => !m.error);
          setModels(available);
          if (available.length > 0) setSelectedModel(available[0].id);
          setError(null);
          setModelsLoading(false);
          return;
        } catch (err) {
          if (attempt < retries) await new Promise(r => setTimeout(r, 3000));
          else {
            setError('Could not load models from backend. Is the server running on localhost:8000?');
            setModelsLoading(false);
          }
        }
      }
    };
    fetchModels();
  }, []);

  useEffect(() => {
    if (selectedModel) analyzeText(inputText);
  }, [selectedModel]);

  useEffect(() => {
    if (!selectedModel || activeTab !== TAB_SAE) return;
    const timer = setTimeout(() => analyzeText(inputText), 250);
    return () => clearTimeout(timer);
  }, [topK, selectedModel, activeTab]);

  useEffect(() => {
    return () => {
      if (hoverClearTimerRef.current) clearTimeout(hoverClearTimerRef.current);
    };
  }, []);

  const analyzeText = async (textToAnalyze) => {
    if (!selectedModel) { setError('Please select a model first'); return; }
    setLoading(true);
    setError(null);
    setActiveTokenIndex(null);
    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToAnalyze, model_id: selectedModel, analyzer: 'sae', top_k: topK }),
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Network response was not ok');
      }
      const data = await response.json();
      setTokens(data.tokens);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const runFeatureLookup = async () => {
    if (!selectedModel) { setFeatureLookupError('Please select a model first.'); return; }
    const parsedFeatureId = Number(lookupFeatureId);
    if (!Number.isInteger(parsedFeatureId) || parsedFeatureId < 0) {
      setFeatureLookupError('Enter a valid non-negative feature ID.');
      return;
    }
    const parsedMinActivation = lookupMinActivation.trim() === '' ? 0 : Number(lookupMinActivation);
    if (!Number.isFinite(parsedMinActivation) || parsedMinActivation < 0) {
      setFeatureLookupError('Min activation must be a valid non-negative number.');
      return;
    }
    setFeatureLookupLoading(true);
    setFeatureLookupError(null);
    try {
      const response = await fetch(`${API_BASE}/feature-activations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: selectedModel,
          feature_id: parsedFeatureId,
          dataset_name: 'MLCommons/peoples_speech',
          dataset_config: 'validation',
          split: 'validation',
          max_sentences: Math.max(1, Number(lookupMaxSentences) || 200),
          max_results: Math.max(1, Number(lookupMaxResults) || 100),
          min_activation: parsedMinActivation,
        }),
      });
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Feature activation lookup failed');
      }
      setFeatureLookupResults(await response.json());
    } catch (e) {
      setFeatureLookupError(e.message);
    } finally {
      setFeatureLookupLoading(false);
    }
  };

  const handleTokenClick = (index) => {
    if (hoverClearTimerRef.current) {
      clearTimeout(hoverClearTimerRef.current);
      hoverClearTimerRef.current = null;
    }
    setHoveredTokenIndex(null);
    setActiveTokenIndex(prev => prev === index ? null : index);
  };

  const handleTokenMouseEnter = (index) => {
    if (activeTokenIndex !== null) return;
    if (hoverClearTimerRef.current) {
      clearTimeout(hoverClearTimerRef.current);
      hoverClearTimerRef.current = null;
    }
    setHoveredTokenIndex(index);
  };

  const handleTokenMouseLeave = () => {
    if (activeTokenIndex !== null) return;
    if (hoverClearTimerRef.current) clearTimeout(hoverClearTimerRef.current);
    // Small delay avoids hover thrashing when pointer sits on token borders.
    hoverClearTimerRef.current = setTimeout(() => {
      setHoveredTokenIndex(null);
      hoverClearTimerRef.current = null;
    }, 60);
  };

  const getActiveToken = () => {
    if (hoveredTokenIndex !== null && tokens[hoveredTokenIndex]) return tokens[hoveredTokenIndex];
    if (activeTokenIndex !== null && tokens[activeTokenIndex]) return tokens[activeTokenIndex];
    return null;
  };

  const activeToken = getActiveToken();
  const visibleFeatureCount = topK > 0 ? topK : null;
  const visibleActiveFeatures = activeToken
    ? (visibleFeatureCount ? activeToken.features.slice(0, visibleFeatureCount) : activeToken.features)
    : [];

  const runSentenceFeatureTrace = async () => {
    if (!selectedModel) {
      setTraceError('Please select a model first.');
      return;
    }
    if (!traceText.trim()) {
      setTraceError('Enter a sentence to trace.');
      return;
    }

    const parsedFeatureId = Number(traceFeatureId);
    if (!Number.isInteger(parsedFeatureId) || parsedFeatureId < 0) {
      setTraceError('Enter a valid non-negative feature ID.');
      return;
    }

    const parsedMinActivation = traceMinActivation.trim() === '' ? 0 : Number(traceMinActivation);
    if (!Number.isFinite(parsedMinActivation) || parsedMinActivation < 0) {
      setTraceError('Min activation must be a valid non-negative number.');
      return;
    }

    setTraceLoading(true);
    setTraceError(null);
    setTraceHoveredIndex(null);
    try {
      const response = await fetch(`${API_BASE}/sentence-feature-trace`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: traceText,
          model_id: selectedModel,
          feature_id: parsedFeatureId,
          min_activation: parsedMinActivation,
          max_length: 512,
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Sentence feature trace failed');
      }
      setTraceResult(await response.json());
    } catch (e) {
      setTraceError(e.message);
      setTraceResult(null);
    } finally {
      setTraceLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      {/* Scanline overlay for cyberpunk effect */}
      {Platform.OS === 'web' && themeMode === 'dark' && (
        <View style={styles.scanlineOverlay} pointerEvents="none" />
      )}

      <ScrollView style={styles.scrollContainer} showsVerticalScrollIndicator={false}>
        {/* HEADER */}
        <View style={styles.header}>
          <View style={styles.headerAccent} />
          <View style={styles.headerContent}>
            {/* <Text style={styles.headerEyebrow}>//</Text> */}
            <Text style={styles.headerTitle}>Feature Interpretation using SAE</Text>
            <Text style={styles.headerSub}>by ANTLR</Text>
          </View>
          <View style={styles.themeSwitchWrap}>
            <TouchableOpacity
              style={[styles.themeSwitchBtn, themeMode === 'dark' && styles.themeSwitchBtnActive]}
              onPress={() => setThemeMode('dark')}
            >
              <Text style={[styles.themeSwitchText, themeMode === 'dark' && styles.themeSwitchTextActive]}>DARK</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.themeSwitchBtn, themeMode === 'light' && styles.themeSwitchBtnActive]}
              onPress={() => setThemeMode('light')}
            >
              <Text style={[styles.themeSwitchText, themeMode === 'light' && styles.themeSwitchTextActive]}>WHITE</Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* TAB BAR */}
        <View style={styles.tabBar}>
          {[
            { id: TAB_SAE, label: 'SAE ANALYSIS', icon: '⬡' },
            { id: TAB_FEATURE_MAP, label: 'FEATURE MAP', icon: '◎' },
            { id: TAB_FEATURE_TRACE, label: 'FEATURE TRACE', icon: '◍' },
          ].map(tab => (
            <TouchableOpacity
              key={tab.id}
              style={[styles.tabItem, activeTab === tab.id && styles.tabItemActive]}
              onPress={() => setActiveTab(tab.id)}
            >
              <Text style={styles.tabIcon}>{tab.icon}</Text>
              <Text style={[styles.tabText, activeTab === tab.id && styles.tabTextActive]}>{tab.label}</Text>
              {activeTab === tab.id && <View style={styles.tabUnderline} />}
            </TouchableOpacity>
          ))}
        </View>

        {/* ===================== SAE TAB ===================== */}
        {activeTab === TAB_SAE && (
          <View style={[styles.contentContainer, isLargeScreen ? styles.row : styles.column]}>

            {/* LEFT PANEL */}
            <View style={[styles.panel, isLargeScreen && styles.leftPanelLarge]}>
              {/* Model Selector */}
              <View style={styles.sectionBlock}>
                <View style={styles.sectionLabelRow}>
                  <View style={styles.sectionDot} />
                  <Text style={styles.sectionLabel}>SAE CHECKPOINT</Text>
                </View>
                <View style={styles.pickerWrapper}>
                  {modelsLoading ? (
                    <ActivityIndicator color="#00ffcc" style={{ padding: 12 }} />
                  ) : (
                    <Picker
                      selectedValue={selectedModel}
                      onValueChange={setSelectedModel}
                      style={styles.picker}
                      itemStyle={styles.pickerItem}
                    >
                      {models.map((m) => (
                        <Picker.Item key={m.id} label={`${m.name}  [${m.d_hidden || '?'} features]`} value={m.id} />
                      ))}
                    </Picker>
                  )}
                </View>
              </View>

              {/* Input */}
              <View style={styles.sectionBlock}>
                <View style={styles.sectionLabelRow}>
                  <View style={styles.sectionDot} />
                  <Text style={styles.sectionLabel}>INPUT CORPUS</Text>
                </View>
                <View style={styles.inputCard}>
                  <TextInput
                    style={styles.textInput}
                    multiline
                    value={inputText}
                    onChangeText={setInputText}
                    placeholder="// enter text to analyze..."
                    placeholderTextColor={C.textMuted}
                  />
                </View>

                <View style={styles.analyzeRow}>
                  <TouchableOpacity
                    style={[styles.analyzeButton, loading && styles.analyzeButtonDisabled]}
                    onPress={() => analyzeText(inputText)}
                    disabled={loading}
                  >
                    {loading
                      ? <ActivityIndicator color="#000" size="small" />
                      : <Text style={styles.analyzeButtonText}>▶ ANALYZE</Text>
                    }
                  </TouchableOpacity>
                  <View style={styles.topKRow}>
                    <Text style={styles.topKLabel}>TOP K</Text>
                    <TextInput
                      style={styles.topKInput}
                      value={String(topK)}
                      onChangeText={(t) => {
                        const d = t.replace(/[^0-9]/g, '');
                        setTopK(d === '' ? 0 : Number(d));
                      }}
                      keyboardType="numeric"
                      maxLength={4}
                    />
                    <Text style={styles.topKHint}>{topK === 0 ? 'ALL' : `features`}</Text>
                  </View>
                </View>
              </View>

              {/* Token Display */}
              <View style={styles.sectionBlock}>
                <View style={styles.sectionLabelRow}>
                  <View style={styles.sectionDot} />
                  <Text style={styles.sectionLabel}>TOKEN VIEW</Text>
                  <Text style={styles.sectionSub}>— hover or tap a token to inspect</Text>
                </View>
                <View style={styles.tokenContainer}>
                  <Text style={styles.tokenWrapper}>
                    {tokens.length === 0
                      ? <Text style={styles.tokenPlaceholder}>{'// awaiting input...'}</Text>
                      : tokens.map((token, index) => {
                          const isActive = activeTokenIndex === index || hoveredTokenIndex === index;
                          const hasFeatures = (token.features || []).length > 0;
                          return (
                            <Text
                              key={index}
                              style={[
                                styles.tokenText,
                                hasFeatures && styles.tokenHasFeatures,
                                isActive && styles.tokenActive,
                              ]}
                              onPress={() => handleTokenClick(index)}
                              {...(Platform.OS === 'web' ? {
                                onMouseEnter: () => handleTokenMouseEnter(index),
                                onMouseLeave: () => handleTokenMouseLeave(),
                              } : {})}
                            >
                              {token.text}
                            </Text>
                          );
                        })}
                  </Text>
                </View>
              </View>
            </View>

            {/* RIGHT PANEL */}
            <View style={[styles.panel, isLargeScreen && styles.rightPanelLarge]}>
              <View style={styles.sectionLabelRow}>
                <View style={[styles.sectionDot, { backgroundColor: '#ff00cc' }]} />
                <Text style={styles.sectionLabel}>ACTIVATED FEATURES</Text>
                {activeToken && (
                  <TouchableOpacity style={styles.clearBtn} onPress={() => setActiveTokenIndex(null)}>
                    <Text style={styles.clearBtnText}>✕ CLEAR</Text>
                  </TouchableOpacity>
                )}
              </View>

              {activeToken ? (
                <View>
                  <View style={styles.selectedTokenChip}>
                    <Text style={styles.selectedTokenGlyph}>◈</Text>
                    <Text style={styles.selectedTokenText}>{activeToken.text.trim() || '(space)'}</Text>
                    <Text style={styles.selectedTokenCount}>{visibleActiveFeatures.length} / {activeToken.features.length} features</Text>
                  </View>

                  {visibleActiveFeatures.length === 0 ? (
                    <View style={styles.emptyState}>
                      <Text style={styles.emptyStateText}>No active features for this token.</Text>
                    </View>
                  ) : (
                    <View>
                      {visibleActiveFeatures.map((feature) => (
                        <FeatureChip
                          key={feature.id}
                          feature={feature}
                          styles={styles}
                          onPress={() => setSelectedFeature(feature)}
                        />
                      ))}
                    </View>
                  )}
                </View>
              ) : (
                <View style={styles.rightPanelPlaceholder}>
                  <Text style={styles.placeholderGlyph}>◎</Text>
                  <Text style={styles.placeholderTitle}>SELECT A TOKEN</Text>
                  <Text style={styles.placeholderSub}>
                    Tap any highlighted word in the token view to inspect its activated SAE features.
                  </Text>
                </View>
              )}

              {error && (
                <View style={styles.errorBox}>
                  <Text style={styles.errorLabel}>⚠ BACKEND ERROR</Text>
                  <Text style={styles.errorText}>{error}</Text>
                </View>
              )}
            </View>
          </View>
        )}

        {/* ===================== FEATURE MAP TAB ===================== */}
        {activeTab === TAB_FEATURE_MAP && (
          <View style={styles.contentContainer}>
            <View style={styles.panel}>

              {/* Feature Lookup */}
              <View style={styles.sectionBlock}>
                <View style={styles.sectionLabelRow}>
                  <View style={[styles.sectionDot, { backgroundColor: '#ffcc00' }]} />
                  <Text style={styles.sectionLabel}>DATASET ACTIVATION SEARCH</Text>
                </View>
                <Text style={styles.lookupHint}>
                  Find sentences in MLCommons/peoples_speech where a specific feature activates.
                </Text>

                <View style={styles.lookupGrid}>
                  {[
                    { label: 'FEATURE ID', value: lookupFeatureId, onChange: (t) => setLookupFeatureId(t.replace(/[^0-9]/g, '')), placeholder: 'e.g. 42' },
                    { label: 'MAX SENTENCES', value: lookupMaxSentences, onChange: (t) => setLookupMaxSentences(t.replace(/[^0-9]/g, '')), placeholder: '200' },
                    { label: 'MAX RESULTS', value: lookupMaxResults, onChange: (t) => setLookupMaxResults(t.replace(/[^0-9]/g, '')), placeholder: '100' },
                    { label: 'MIN ACTIVATION', value: lookupMinActivation, onChange: (t) => setLookupMinActivation(sanitizeDecimalInput(t)), placeholder: '0.0' },
                  ].map(field => (
                    <View key={field.label} style={styles.lookupField}>
                      <Text style={styles.lookupFieldLabel}>{field.label}</Text>
                      <TextInput
                        style={styles.lookupInput}
                        value={field.value}
                        onChangeText={field.onChange}
                        keyboardType="numeric"
                        placeholder={field.placeholder}
                        placeholderTextColor={C.textMuted}
                      />
                    </View>
                  ))}
                </View>

                <View style={styles.lookupActionsRow}>
                  <TouchableOpacity
                    style={[styles.analyzeButton, featureLookupLoading && styles.analyzeButtonDisabled]}
                    onPress={runFeatureLookup}
                    disabled={featureLookupLoading}
                  >
                    {featureLookupLoading
                      ? <ActivityIndicator color="#000" size="small" />
                      : <Text style={styles.analyzeButtonText}>▶ FIND ACTIVATIONS</Text>
                    }
                  </TouchableOpacity>

                  {!!selectedFeature?.id && (
                    <TouchableOpacity
                      style={styles.secondaryButton}
                      onPress={() => setLookupFeatureId(String(selectedFeature.id))}
                    >
                      <Text style={styles.secondaryButtonText}>USE FEATURE #{selectedFeature.id}</Text>
                    </TouchableOpacity>
                  )}
                </View>

                {featureLookupError && (
                  <View style={styles.errorBox}>
                    <Text style={styles.errorLabel}>⚠ ERROR</Text>
                    <Text style={styles.errorText}>{featureLookupError}</Text>
                  </View>
                )}

                {featureLookupResults && (
                  <View style={styles.lookupResults}>
                    <View style={styles.lookupResultsHeader}>
                      <Text style={styles.lookupResultsTitle}>FEATURE #{featureLookupResults.feature_id}</Text>
                      <Text style={styles.lookupResultsDesc}>{featureLookupResults.feature_description}</Text>
                      <View style={styles.lookupResultsMeta}>
                        <Text style={styles.lookupResultsMetaText}>
                          Scanned {featureLookupResults.scanned_sentences} sentences · {featureLookupResults.matches?.length ?? 0} matches shown
                        </Text>
                      </View>
                    </View>

                    {(featureLookupResults.matches || []).map((match, idx) => (
                      <View key={idx} style={styles.matchCard}>
                        <View style={styles.matchCardHeader}>
                          <Text style={styles.matchCardMeta}>
                            Sentence #{match.sentence_index} · Token #{match.token_index}
                          </Text>
                          <View style={styles.matchActivationBadge}>
                            <Text style={styles.matchActivationText}>{match.activation?.toFixed(4)}</Text>
                          </View>
                        </View>
                        <Text style={styles.matchContext}>
                          {match.left_context}
                          <Text style={styles.matchToken}>{match.token}</Text>
                          {match.right_context}
                        </Text>
                      </View>
                    ))}
                  </View>
                )}
              </View>

            </View>
          </View>
        )}

        {/* ===================== FEATURE TRACE TAB ===================== */}
        {activeTab === TAB_FEATURE_TRACE && (
          <View style={styles.contentContainer}>
            <View style={[styles.traceGrid, isLargeScreen ? styles.row : styles.column]}>
              <View style={[styles.panel, isLargeScreen && styles.leftPanelLarge]}>
                <View style={styles.sectionBlock}>
                  <View style={styles.sectionLabelRow}>
                    <View style={[styles.sectionDot, { backgroundColor: C.cyan }]} />
                    <Text style={styles.sectionLabel}>TRACE INPUT</Text>
                  </View>

                  <View style={styles.inputCard}>
                    <TextInput
                      style={styles.textInput}
                      multiline
                      value={traceText}
                      onChangeText={setTraceText}
                      placeholder="// enter sentence to trace..."
                      placeholderTextColor={C.textMuted}
                    />
                  </View>

                  <View style={styles.lookupGrid}>
                    <View style={styles.lookupField}>
                      <Text style={styles.lookupFieldLabel}>FEATURE ID</Text>
                      <TextInput
                        style={styles.lookupInput}
                        value={traceFeatureId}
                        onChangeText={(t) => setTraceFeatureId(t.replace(/[^0-9]/g, ''))}
                        keyboardType="numeric"
                        placeholder="e.g. 42"
                        placeholderTextColor={C.textMuted}
                      />
                    </View>
                    <View style={styles.lookupField}>
                      <Text style={styles.lookupFieldLabel}>MIN ACTIVATION</Text>
                      <TextInput
                        style={styles.lookupInput}
                        value={traceMinActivation}
                        onChangeText={(t) => setTraceMinActivation(sanitizeDecimalInput(t))}
                        keyboardType="numeric"
                        placeholder="0.0"
                        placeholderTextColor={C.textMuted}
                      />
                    </View>
                  </View>

                  <View style={styles.lookupActionsRow}>
                    <TouchableOpacity
                      style={[styles.analyzeButton, traceLoading && styles.analyzeButtonDisabled]}
                      onPress={runSentenceFeatureTrace}
                      disabled={traceLoading}
                    >
                      {traceLoading
                        ? <ActivityIndicator color="#000" size="small" />
                        : <Text style={styles.analyzeButtonText}>▶ TRACE FEATURE</Text>
                      }
                    </TouchableOpacity>
                  </View>
                </View>

                {traceError && (
                  <View style={styles.errorBox}>
                    <Text style={styles.errorLabel}>⚠ ERROR</Text>
                    <Text style={styles.errorText}>{traceError}</Text>
                  </View>
                )}

                {traceResult && (
                  <View style={styles.sectionBlock}>
                    <View style={styles.lookupResultsHeader}>
                      <Text style={styles.lookupResultsTitle}>FEATURE #{traceResult.feature_id}</Text>
                      <Text style={styles.lookupResultsDesc}>{traceResult.feature_name}</Text>
                      <Text style={styles.lookupResultsMetaText}>{traceResult.feature_description}</Text>
                    </View>
                    <View style={styles.traceStatsRow}>
                      <View style={styles.metaChip}>
                        <Text style={styles.metaChipLabel}>ACTIVE TOKENS</Text>
                        <Text style={styles.metaChipValue}>{traceResult.active_token_count}</Text>
                      </View>
                      <View style={styles.metaChip}>
                        <Text style={styles.metaChipLabel}>TOKEN COUNT</Text>
                        <Text style={styles.metaChipValue}>{traceResult.token_count}</Text>
                      </View>
                      <View style={styles.metaChip}>
                        <Text style={styles.metaChipLabel}>MAX ACT</Text>
                        <Text style={styles.metaChipValue}>{Number(traceResult.max_activation || 0).toFixed(3)}</Text>
                      </View>
                      <View style={styles.metaChip}>
                        <Text style={styles.metaChipLabel}>THRESHOLD</Text>
                        <Text style={styles.metaChipValue}>{Number(traceResult.min_activation || 0).toFixed(3)}</Text>
                      </View>
                    </View>
                  </View>
                )}
              </View>

              <View style={[styles.panel, isLargeScreen && styles.rightPanelLarge]}>
                <View style={styles.sectionLabelRow}>
                  <View style={[styles.sectionDot, { backgroundColor: '#ff00cc' }]} />
                  <Text style={styles.sectionLabel}>TOKEN ACTIVATION MAP</Text>
                </View>

                {!traceResult ? (
                  <View style={styles.emptyState}>
                    <Text style={styles.placeholderGlyph}>◎</Text>
                    <Text style={styles.emptyStateText}>Run a trace to view token-level activations for the selected feature.</Text>
                  </View>
                ) : (
                  <View style={styles.traceSentenceCard}>
                    <Text style={styles.traceSentenceText}>
                      {(traceResult.tokens || []).map((tok, idx) => {
                        const isHovered = traceHoveredIndex === idx;
                        return (
                          <Text
                            key={`${idx}-${tok.index}`}
                            style={[
                              styles.traceToken,
                              tok.is_active && styles.traceTokenActive,
                              isHovered && styles.traceTokenHovered,
                            ]}
                            onPress={() => setTraceHoveredIndex(isHovered ? null : idx)}
                            {...(Platform.OS === 'web' ? {
                              onMouseEnter: () => setTraceHoveredIndex(idx),
                              onMouseLeave: () => setTraceHoveredIndex(null),
                            } : {})}
                          >
                            {tok.text}
                          </Text>
                        );
                      })}
                    </Text>
                  </View>
                )}

                {traceResult && traceHoveredIndex !== null && traceResult.tokens?.[traceHoveredIndex] && (
                  <View style={styles.traceHoverCard}>
                    <Text style={styles.traceHoverTitle}>TOKEN #{traceResult.tokens[traceHoveredIndex].index}</Text>
                    <Text style={styles.traceHoverMeta}>
                      Activation: {Number(traceResult.tokens[traceHoveredIndex].activation || 0).toFixed(6)}
                    </Text>
                    <Text style={styles.traceHoverMeta}>
                      Active: {traceResult.tokens[traceHoveredIndex].is_active ? 'yes' : 'no'}
                    </Text>
                  </View>
                )}
              </View>
            </View>
          </View>
        )}
      </ScrollView>

      {/* FOOTER */}
      <View style={styles.footer}>
        <View style={styles.footerInner}>
          <Text style={styles.footerGlyph}>◈</Text>
          <Text style={styles.footerText}>Nafis Nahian · Arnob Biswas · Tanvir Liaquat Uday</Text>
          <Text style={styles.footerGlyph}>◈</Text>
        </View>
      </View>

      {/* Feature Details Modal */}
      <Modal
        animationType="slide"
        transparent={false}
        visible={!!selectedFeature}
        onRequestClose={() => setSelectedFeature(null)}
      >
        <FeatureDetails
          feature={selectedFeature}
          onClose={() => setSelectedFeature(null)}
          modelId={selectedModel}
          themeMode={themeMode}
        />
      </Modal>

      <StatusBar style={themeMode === 'light' ? 'dark' : 'light'} />
    </View>
  );
}

// ─── CYBERPUNK COLOR TOKENS ───────────────────────────────────────────────────
const DARK_THEME = {
  bg:          '#06090f',
  bgPanel:     '#0d1117',
  bgCard:      '#111827',
  bgCardHover: '#1a2435',
  border:      '#1e2d3d',
  borderAccent:'#00ffcc',
  borderPink:  '#ff00cc',
  borderYellow:'#ffcc00',
  cyan:        '#00ffcc',
  pink:        '#ff00cc',
  yellow:      '#ffcc00',
  blue:        '#00aaff',
  textPrimary: '#e0f0ff',
  textSecond:  '#9ab3cc',
  textMuted:   '#7e95ad',
  error:       '#ff3355',
  errorBg:     '#1a0010',
};

const LIGHT_THEME = {
  bg:          '#f1f1f1',
  bgPanel:     '#ffffff',
  bgCard:      '#fbfbfb',
  bgCardHover: '#f2f2f2',
  border:      '#d8d8d8',
  borderAccent:'#e3a55d',
  borderPink:  '#5f79c9',
  borderYellow:'#e3a55d',
  cyan:        '#d89b54',
  pink:        '#5f79c9',
  yellow:      '#d89b54',
  blue:        '#4a67bf',
  textPrimary: '#101317',
  textSecond:  '#1c222a',
  textMuted:   '#383f49',
  error:       '#c54141',
  errorBg:     '#fff1f1',
};

const createStyles = (C) => StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: C.bg,
  },
  scanlineOverlay: {
    position: 'absolute',
    top: 0, left: 0, right: 0, bottom: 0,
    backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,204,0.015) 2px, rgba(0,255,204,0.015) 4px)',
    pointerEvents: 'none',
    zIndex: 9999,
  },
  scrollContainer: { flex: 1 },

  // Header
  header: {
    borderBottomWidth: 1,
    borderBottomColor: C.borderAccent,
    padding: 24,
    backgroundColor: C.bgPanel,
    flexDirection: 'row',
    alignItems: 'center',
    position: 'relative',
    overflow: 'hidden',
  },
  headerAccent: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: 4,
    height: '100%',
    backgroundColor: C.cyan,
  },
  headerContent: { flex: 1, paddingLeft: 12 },
  themeSwitchWrap: {
    flexDirection: 'row',
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 4,
    overflow: 'hidden',
    marginRight: 16,
    backgroundColor: C.bgCard,
  },
  themeSwitchBtn: {
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  themeSwitchBtnActive: {
    backgroundColor: C.cyan,
  },
  themeSwitchText: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 9,
    letterSpacing: 1,
    fontWeight: '700',
    color: C.textMuted,
  },
  themeSwitchTextActive: {
    color: '#000',
  },
  headerEyebrow: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    color: C.cyan,
    letterSpacing: 3,
    marginBottom: 4,
  },
  headerTitle: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 26,
    fontWeight: '900',
    color: C.textPrimary,
    letterSpacing: 4,
    marginBottom: 4,
  },
  headerSub: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 11,
    color: C.textSecond,
    letterSpacing: 2,
  },
  // Tab bar
  tabBar: {
    flexDirection: 'row',
    backgroundColor: C.bgPanel,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
  },
  tabItem: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    gap: 8,
    position: 'relative',
  },
  tabItemActive: {},
  tabIcon: { fontSize: 14, color: C.textSecond },
  tabText: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 11,
    fontWeight: '700',
    color: C.textSecond,
    letterSpacing: 2,
  },
  tabTextActive: { color: C.cyan },
  tabUnderline: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: 2,
    backgroundColor: C.cyan,
  },

  // Layout
  contentContainer: { padding: 20 },
  row: { flexDirection: 'row', alignItems: 'flex-start' },
  column: { flexDirection: 'column' },
  panel: { flex: 1 },
  leftPanelLarge: { flex: 6, marginRight: 16 },
  rightPanelLarge: { flex: 4, marginLeft: 16 },

  // Section blocks
  sectionBlock: { marginBottom: 20 },
  sectionLabelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    gap: 8,
  },
  sectionDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: C.cyan,
  },
  sectionLabel: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    fontWeight: '700',
    color: C.cyan,
    letterSpacing: 3,
  },
  sectionSub: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    color: C.textMuted,
  },

  // Picker
  pickerWrapper: {
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 4,
    backgroundColor: C.bgCard,
    overflow: 'hidden',
  },
  picker: {
    height: 44,
    color: C.textPrimary,
    backgroundColor: 'transparent',
  },
  pickerItem: {
    color: C.textPrimary,
    backgroundColor: C.bgCard,
    fontFamily: 'monospace',
    fontSize: 13,
  },

  // Input
  inputCard: {
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 4,
    padding: 12,
    marginBottom: 10,
  },
  textInput: {
    minHeight: 80,
    fontSize: 14,
    color: C.textPrimary,
    textAlignVertical: 'top',
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    outlineStyle: 'none',
    lineHeight: 22,
  },
  corpusDisplayText: {
    fontSize: 14,
    color: C.textSecond,
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    lineHeight: 22,
  },

  // Analyze row
  analyzeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    flexWrap: 'wrap',
  },
  analyzeButton: {
    backgroundColor: C.cyan,
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 2,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  analyzeButtonDisabled: { opacity: 0.5 },
  analyzeButtonText: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontWeight: '900',
    fontSize: 12,
    color: '#000',
    letterSpacing: 2,
  },
  topKRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  topKLabel: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    color: C.textSecond,
    letterSpacing: 2,
  },
  topKInput: {
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 2,
    backgroundColor: C.bgCard,
    color: C.cyan,
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 14,
    width: 56,
    textAlign: 'center',
    paddingVertical: 6,
  },
  topKHint: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    color: C.textMuted,
  },

  // Token display
  tokenContainer: {
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 4,
    padding: 16,
    minHeight: 80,
  },
  tokenWrapper: { flexDirection: 'row', flexWrap: 'wrap' },
  tokenPlaceholder: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 13,
    color: C.textMuted,
  },
  tokenText: {
    fontSize: 16,
    lineHeight: 30,
    color: C.textSecond,
    paddingHorizontal: 1,
    paddingVertical: 2,
    borderRadius: 2,
    fontFamily: Platform.OS === 'web' ? 'Georgia, serif' : undefined,
  },
  tokenHasFeatures: {
    color: C.textPrimary,
    borderBottomWidth: 2,
    borderBottomColor: C.borderAccent + '66',
  },
  tokenActive: {
    backgroundColor: C.cyan,
    color: '#000',
    fontWeight: '700',
    borderBottomWidth: 0,
  },

  // Right panel
  clearBtn: {
    marginLeft: 'auto',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderWidth: 1,
    borderColor: C.error,
    borderRadius: 2,
  },
  clearBtnText: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 9,
    color: C.error,
    letterSpacing: 1,
  },
  selectedTokenChip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.cyan,
    borderRadius: 2,
    paddingHorizontal: 12,
    paddingVertical: 8,
    marginBottom: 16,
    gap: 10,
  },
  selectedTokenGlyph: {
    fontSize: 14,
    color: C.cyan,
  },
  selectedTokenText: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 16,
    fontWeight: '700',
    color: C.textPrimary,
    flex: 1,
  },
  selectedTokenCount: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    color: C.textSecond,
  },

  // Feature Card
  featureCard: {
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.border,
    borderLeftWidth: 3,
    borderLeftColor: C.cyan,
    borderRadius: 4,
    padding: 12,
    marginBottom: 8,
  },
  featureCardHovered: {
    backgroundColor: C.bgCardHover,
    borderColor: C.cyan + '88',
    borderLeftColor: C.pink,
  },
  featureCardCompact: {
    padding: 8,
  },
  featureCardTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  featureIdBadge: {
    backgroundColor: C.cyan + '22',
    borderWidth: 1,
    borderColor: C.cyan + '55',
    borderRadius: 2,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  featureIdText: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    fontWeight: '700',
    color: C.cyan,
    letterSpacing: 1,
  },
  activationPill: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 3,
  },
  activationValue: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 14,
    fontWeight: '700',
    color: C.yellow,
  },
  activationUnit: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 8,
    color: C.textMuted,
    letterSpacing: 1,
  },
  featureLabel: {
    fontSize: 13,
    color: C.textSecond,
    lineHeight: 18,
    fontFamily: Platform.OS === 'web' ? 'Georgia, serif' : undefined,
  },

  // Tooltip
  tooltipLeft: {
    position: 'absolute',
    top: 0,
    right: '100%',
    width: 280,
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.cyan,
    borderRadius: 4,
    padding: 12,
    zIndex: 100,
    shadowColor: C.cyan,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 8,
    marginRight: 8,
  },
  tooltipTitle: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    fontWeight: '700',
    color: C.cyan,
    letterSpacing: 2,
    marginBottom: 6,
  },
  tooltipDesc: {
    fontSize: 13,
    color: C.textPrimary,
    lineHeight: 20,
    marginBottom: 6,
    fontFamily: Platform.OS === 'web' ? 'Georgia, serif' : undefined,
  },
  tooltipHint: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 9,
    color: C.pink,
    letterSpacing: 1,
  },

  // Small chips
  smallChip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 2,
    paddingHorizontal: 8,
    paddingVertical: 4,
    gap: 6,
  },
  smallChipHovered: {
    borderColor: C.cyan,
    backgroundColor: C.bgCardHover,
  },
  smallChipId: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 9,
    fontWeight: '700',
    color: C.cyan,
  },
  smallChipLabel: {
    fontSize: 11,
    color: C.textSecond,
    maxWidth: 120,
  },

  // Right panel placeholder
  rightPanelPlaceholder: {
    borderWidth: 1,
    borderColor: C.border,
    borderStyle: 'dashed',
    borderRadius: 4,
    padding: 40,
    alignItems: 'center',
    marginTop: 10,
  },
  placeholderGlyph: {
    fontSize: 36,
    color: C.textMuted,
    marginBottom: 12,
  },
  placeholderTitle: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 12,
    fontWeight: '700',
    color: C.textSecond,
    letterSpacing: 3,
    marginBottom: 10,
  },
  placeholderSub: {
    fontSize: 13,
    color: C.textMuted,
    textAlign: 'center',
    lineHeight: 20,
    maxWidth: 280,
  },
  emptyState: {
    padding: 30,
    alignItems: 'center',
  },
  emptyStateText: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 12,
    color: C.textMuted,
    textAlign: 'center',
  },

  // Error
  errorBox: {
    backgroundColor: C.errorBg,
    borderWidth: 1,
    borderColor: C.error + '66',
    borderRadius: 4,
    padding: 12,
    marginTop: 12,
  },
  errorLabel: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 9,
    fontWeight: '700',
    color: C.error,
    letterSpacing: 2,
    marginBottom: 4,
  },
  errorText: {
    fontSize: 13,
    color: C.error + 'cc',
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
  },

  // Feature Map
  metaChipsRow: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 10,
  },
  metaChip: {
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 2,
    paddingHorizontal: 12,
    paddingVertical: 8,
    alignItems: 'center',
  },
  metaChipLabel: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 8,
    color: C.textMuted,
    letterSpacing: 2,
    marginBottom: 2,
  },
  metaChipValue: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 18,
    fontWeight: '700',
    color: C.cyan,
  },

  // Lookup
  lookupHint: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 11,
    color: C.textSecond,
    marginBottom: 12,
    lineHeight: 18,
  },
  lookupGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 12,
  },
  lookupField: {
    minWidth: 140,
    flex: 1,
  },
  lookupFieldLabel: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 9,
    color: C.textSecond,
    letterSpacing: 2,
    marginBottom: 6,
  },
  lookupInput: {
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 2,
    paddingHorizontal: 10,
    paddingVertical: 8,
    fontSize: 14,
    color: C.cyan,
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    outlineStyle: 'none',
  },
  lookupActionsRow: {
    flexDirection: 'row',
    gap: 10,
    alignItems: 'center',
    flexWrap: 'wrap',
    marginBottom: 12,
  },
  traceGrid: {
    gap: 16,
  },
  traceStatsRow: {
    flexDirection: 'row',
    gap: 10,
    flexWrap: 'wrap',
    marginTop: 10,
  },
  traceSentenceCard: {
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 4,
    padding: 14,
  },
  traceSentenceText: {
    fontSize: 17,
    lineHeight: 30,
    color: C.textSecond,
    fontFamily: Platform.OS === 'web' ? 'Georgia, serif' : undefined,
  },
  traceToken: {
    color: C.textSecond,
    borderRadius: 2,
    paddingHorizontal: 1,
    paddingVertical: 2,
  },
  traceTokenActive: {
    color: C.textPrimary,
    backgroundColor: C.cyan + '22',
    borderBottomWidth: 2,
    borderBottomColor: C.cyan + '88',
  },
  traceTokenHovered: {
    backgroundColor: C.yellow + '33',
    color: C.yellow,
  },
  traceHoverCard: {
    marginTop: 12,
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.yellow + '88',
    borderRadius: 4,
    padding: 12,
  },
  traceHoverTitle: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    color: C.yellow,
    letterSpacing: 2,
    marginBottom: 6,
  },
  traceHoverMeta: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 12,
    color: C.textPrimary,
    lineHeight: 18,
  },
  secondaryButton: {
    borderWidth: 1,
    borderColor: C.yellow + '88',
    backgroundColor: C.yellow + '11',
    borderRadius: 2,
    paddingVertical: 9,
    paddingHorizontal: 12,
  },
  secondaryButtonText: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    fontWeight: '700',
    color: C.yellow,
    letterSpacing: 1,
  },
  lookupResults: {
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 4,
    overflow: 'hidden',
  },
  lookupResultsHeader: {
    backgroundColor: C.bgCard,
    padding: 14,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
  },
  lookupResultsTitle: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 14,
    fontWeight: '700',
    color: C.cyan,
    marginBottom: 4,
  },
  lookupResultsDesc: {
    fontSize: 13,
    color: C.textPrimary,
    lineHeight: 20,
    marginBottom: 6,
  },
  lookupResultsMeta: {},
  lookupResultsMetaText: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    color: C.textMuted,
  },
  matchCard: {
    backgroundColor: C.bgPanel,
    padding: 12,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
  },
  matchCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  matchCardMeta: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    color: C.textMuted,
  },
  matchActivationBadge: {
    backgroundColor: C.yellow + '22',
    borderWidth: 1,
    borderColor: C.yellow + '55',
    borderRadius: 2,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  matchActivationText: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    fontWeight: '700',
    color: C.yellow,
  },
  matchContext: {
    fontSize: 14,
    color: C.textSecond,
    lineHeight: 22,
  },
  matchToken: {
    backgroundColor: C.cyan + '33',
    color: C.cyan,
    fontWeight: '700',
  },

  // Map token cards
  mapTokenCard: {
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.border,
    borderLeftWidth: 3,
    borderLeftColor: C.cyan,
    borderRadius: 4,
    padding: 12,
    marginBottom: 10,
  },
  mapTokenCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  mapTokenBadge: {
    backgroundColor: C.cyan + '22',
    borderRadius: 2,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  mapTokenBadgeText: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 9,
    fontWeight: '700',
    color: C.cyan,
    letterSpacing: 1,
  },
  mapTokenFeatureCount: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    color: C.textMuted,
  },
  mapContextText: {
    fontSize: 14,
    color: C.textSecond,
    lineHeight: 22,
    marginBottom: 10,
  },
  mapContextHighlight: {
    backgroundColor: C.cyan + '33',
    color: C.cyan,
    fontWeight: '700',
  },
  mapFeatureList: { gap: 4 },
  mapFeatureRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: C.bgPanel,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 2,
    paddingVertical: 7,
    paddingHorizontal: 10,
    gap: 10,
  },
  mapFeatureId: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    fontWeight: '700',
    color: C.cyan,
    minWidth: 44,
  },
  mapFeatureDesc: {
    flex: 1,
    fontSize: 12,
    color: C.textSecond,
  },
  mapFeatureAct: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 11,
    fontWeight: '700',
    color: C.yellow,
  },

  // Feature cards in map
  mapFeatureCard: {
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.border,
    borderLeftWidth: 3,
    borderLeftColor: C.pink,
    borderRadius: 4,
    padding: 12,
    marginBottom: 10,
  },
  mapFeatureCardHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    marginBottom: 6,
  },
  mapFeatureCardId: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 12,
    fontWeight: '700',
    color: C.pink,
    minWidth: 44,
  },
  mapFeatureCardDesc: {
    flex: 1,
    fontSize: 13,
    color: C.textPrimary,
    lineHeight: 20,
  },
  mapFeatureCardMeta: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 9,
    color: C.textMuted,
    letterSpacing: 1,
    marginBottom: 8,
  },
  tokenChipRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  tokenChip: {
    backgroundColor: C.pink + '22',
    borderWidth: 1,
    borderColor: C.pink + '55',
    borderRadius: 2,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  tokenChipText: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    fontWeight: '700',
    color: C.pink,
  },

  // Footer
  footer: {
    borderTopWidth: 1,
    borderTopColor: C.border,
    backgroundColor: C.bgPanel,
    padding: 16,
  },
  footerInner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
  },
  footerGlyph: {
    fontSize: 12,
    color: C.cyan,
    opacity: 0.5,
  },
  footerText: {
    fontFamily: Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace',
    fontSize: 10,
    color: C.textMuted,
    letterSpacing: 2,
  },
});