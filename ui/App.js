import { StatusBar } from 'expo-status-bar';
import { useEffect, useMemo, useState } from 'react';
import { StyleSheet, Text, View, ScrollView, useWindowDimensions, TouchableOpacity, ActivityIndicator, TextInput, Platform, Modal } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import FeatureDetails from './FeatureDetails';

const DEFAULT_TEXT = "Did you know that pineapples were a symbol of hospitality in colonial America? This exotic fruit, once a rare delicacy, was often displayed at gatherings to impress guests.";
const API_BASE = 'http://localhost:8000';

// Tab constants
const TAB_SAE = 'sae';
const TAB_SYNONYM = 'synonym';
const TAB_CAPS = 'caps';
const TAB_FEATURE_MAP = 'feature-map';

export default function App() {
  const [activeTab, setActiveTab] = useState(TAB_SAE);

  const [tokens, setTokens] = useState([]);
  const [inputText, setInputText] = useState(DEFAULT_TEXT);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Model selection
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelsLoading, setModelsLoading] = useState(true);
  
  // Selection State
  const [activeTokenIndex, setActiveTokenIndex] = useState(null);
  const [hoveredTokenIndex, setHoveredTokenIndex] = useState(null);
  const [topK, setTopK] = useState(0); // 0 = all active features for selected token
  const [selectedFeature, setSelectedFeature] = useState(null); // Feature for details modal

  // Synonym test state
  const [synonymClusters, setSynonymClusters] = useState([]);
  const [synonymClusterDetails, setSynonymClusterDetails] = useState({});
  const [selectedSynonymClusters, setSelectedSynonymClusters] = useState([]);
  const [synonymResults, setSynonymResults] = useState(null);
  const [synonymLoading, setSynonymLoading] = useState(false);
  const [customSynonymWords, setCustomSynonymWords] = useState('');
  const [synonymMode, setSynonymMode] = useState('custom'); // 'custom' or 'preset'
  const [synonymTopK, setSynonymTopK] = useState(30);

  // Caps test state
  const [capsWords, setCapsWords] = useState([]);
  const [selectedCapsWords, setSelectedCapsWords] = useState([]);
  const [capsResults, setCapsResults] = useState(null);
  const [capsLoading, setCapsLoading] = useState(false);

  // Feature lookup state (feature -> sentences/tokens from dataset)
  const [lookupFeatureId, setLookupFeatureId] = useState('');
  const [lookupMaxSentences, setLookupMaxSentences] = useState('200');
  const [lookupMaxResults, setLookupMaxResults] = useState('100');
  const [lookupMinActivation, setLookupMinActivation] = useState('0.0');
  const [featureLookupLoading, setFeatureLookupLoading] = useState(false);
  const [featureLookupError, setFeatureLookupError] = useState(null);
  const [featureLookupResults, setFeatureLookupResults] = useState(null);

  const { width } = useWindowDimensions();
  const isLargeScreen = width > 768;

  // Fetch available models on mount
  useEffect(() => {
    const fetchModels = async (retries = 3) => {
      for (let attempt = 1; attempt <= retries; attempt++) {
        try {
          const controller = new AbortController();
          const timeout = setTimeout(() => controller.abort(), 60000); // 60s timeout
          const res = await fetch(`${API_BASE}/models?analyzer=sae`, {
            signal: controller.signal,
          });
          clearTimeout(timeout);
          const data = await res.json();
          const available = (data.models || []).filter(m => !m.error);
          setModels(available);
          if (available.length > 0) {
            setSelectedModel(available[0].id);
          }
          setError(null);
          setModelsLoading(false);
          return; // success
        } catch (err) {
          console.error(`Attempt ${attempt}/${retries} failed:`, err);
          if (attempt < retries) {
            // Wait before retrying (backend may still be loading models)
            await new Promise(r => setTimeout(r, 3000));
          } else {
            setError('Could not load models from backend. Is the server running on localhost:8000?');
            setModelsLoading(false);
          }
        }
      }
    };
    fetchModels();
  }, []);

  // Auto-analyze when model is first set
  useEffect(() => {
    if (selectedModel) {
      analyzeText(inputText);
    }
  }, [selectedModel]);

  // Refresh analysis whenever Top K changes so backend returns matching feature count.
  useEffect(() => {
    if (!selectedModel || activeTab !== TAB_SAE) return;
    const timer = setTimeout(() => {
      analyzeText(inputText);
    }, 250);
    return () => clearTimeout(timer);
  }, [topK, selectedModel, activeTab]);

  // Fetch synonym clusters and caps words metadata
  useEffect(() => {
    const fetchMeta = async () => {
      try {
        const [synRes, capsRes] = await Promise.all([
          fetch(`${API_BASE}/synonym-clusters`).then(r => r.json()).catch(() => null),
          fetch(`${API_BASE}/caps-words`).then(r => r.json()).catch(() => null),
        ]);
        if (synRes?.clusters) setSynonymClusters(synRes.clusters);
        if (synRes?.details) setSynonymClusterDetails(synRes.details);
        if (capsRes?.words) setCapsWords(capsRes.words);
      } catch (e) {
        console.error('Failed to fetch test metadata:', e);
      }
    };
    fetchMeta();
  }, []);

  const analyzeText = async (textToAnalyze) => {
    if (!selectedModel) {
      setError('Please select a model first');
      return;
    }
    setLoading(true);
    setError(null);
    setActiveTokenIndex(null); // Reset selection
    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: textToAnalyze,
          model_id: selectedModel,
          analyzer: 'sae',
          // 0 means "all active features" (handled in backend analyzer).
          top_k: topK,
        }),
      });
      
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Network response was not ok');
      }
      
      const data = await response.json();
      setTokens(data.tokens);
    } catch (err) {
      console.error("Fetch error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ----------- Synonym Test Runner -----------
  const runSynonymTest = async () => {
    if (!selectedModel) { setError('Please select a model first'); return; }
    setSynonymLoading(true);
    setError(null);
    try {
      let body;
      if (synonymMode === 'custom') {
        const words = customSynonymWords.split(',').map(w => w.trim()).filter(Boolean);
        if (words.length < 2) {
          setError('Please enter at least 2 comma-separated words to compare.');
          setSynonymLoading(false);
          return;
        }
        body = {
          model_id: selectedModel,
          custom_words: words,
          top_k: synonymTopK,
        };
      } else {
        body = {
          model_id: selectedModel,
          clusters: selectedSynonymClusters.length > 0 ? selectedSynonymClusters : null,
          top_k: synonymTopK,
        };
      }
      const res = await fetch(`${API_BASE}/synonym-test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Context-similarity test failed');
      }
      const data = await res.json();
      setSynonymResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setSynonymLoading(false);
    }
  };

  // ----------- Caps Test Runner -----------
  const runCapsTest = async () => {
    if (!selectedModel) { setError('Please select a model first'); return; }
    setCapsLoading(true);
    setError(null);
    try {
      const body = {
        model_id: selectedModel,
        words: selectedCapsWords.length > 0 ? selectedCapsWords : null,
        top_k: 30,
      };
      const res = await fetch(`${API_BASE}/caps-test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Caps test failed');
      }
      const data = await res.json();
      setCapsResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setCapsLoading(false);
    }
  };

  const runFeatureLookup = async () => {
    if (!selectedModel) {
      setFeatureLookupError('Please select a model first.');
      return;
    }

    const parsedFeatureId = Number(lookupFeatureId);
    if (!Number.isInteger(parsedFeatureId) || parsedFeatureId < 0) {
      setFeatureLookupError('Enter a valid non-negative feature ID.');
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
          min_activation: Math.max(0, Number(lookupMinActivation) || 0),
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Feature activation lookup failed');
      }

      const data = await response.json();
      setFeatureLookupResults(data);
    } catch (e) {
      setFeatureLookupError(e.message);
    } finally {
      setFeatureLookupLoading(false);
    }
  };

  const handleTokenClick = (index) => {
    if (activeTokenIndex === index) {
      setActiveTokenIndex(null); // Deselect if clicked again
    } else {
      setActiveTokenIndex(index);
    }
  };

  const getActiveToken = () => {
    if (hoveredTokenIndex !== null && tokens[hoveredTokenIndex]) {
      return tokens[hoveredTokenIndex];
    }
    if (activeTokenIndex !== null && tokens[activeTokenIndex]) {
      return tokens[activeTokenIndex];
    }
    return null;
  };

  const activeToken = getActiveToken();
  const visibleFeatureCount = topK > 0 ? topK : null;
  const visibleActiveFeatures = activeToken
    ? (visibleFeatureCount ? activeToken.features.slice(0, visibleFeatureCount) : activeToken.features)
    : [];

  const activatedTokenRows = useMemo(() => {
    return tokens
      .map((token, tokenIndex) => ({ token, tokenIndex }))
      .filter(({ token }) => (token.features || []).length > 0)
      .map(({ token, tokenIndex }) => {
        const features = visibleFeatureCount ? token.features.slice(0, visibleFeatureCount) : token.features;
        const start = Math.max(0, tokenIndex - 5);
        const end = Math.min(tokens.length, tokenIndex + 6);
        const before = tokens.slice(start, tokenIndex).map(t => t.text).join('');
        const after = tokens.slice(tokenIndex + 1, end).map(t => t.text).join('');
        return {
          tokenIndex,
          tokenText: token.text,
          before,
          after,
          totalFeatures: token.features.length,
          features,
        };
      });
  }, [tokens, visibleFeatureCount]);

  const featureToTokensRows = useMemo(() => {
    const featureMap = new Map();
    activatedTokenRows.forEach((row) => {
      row.features.forEach((feature) => {
        if (!featureMap.has(feature.id)) {
          featureMap.set(feature.id, {
            id: feature.id,
            description: feature.description,
            tokens: [],
          });
        }
        featureMap.get(feature.id).tokens.push({
          tokenText: row.tokenText,
          tokenIndex: row.tokenIndex,
          activation: feature.activation,
        });
      });
    });
    return Array.from(featureMap.values()).sort((a, b) => b.tokens.length - a.tokens.length);
  }, [activatedTokenRows]);

  // Helper: signal badge color
  const signalColor = (interpretation) => {
    if (!interpretation) return '#9ca3af';
    if (interpretation.includes('STRONG') || interpretation.includes('INVARIANT')) return '#16a34a';
    if (interpretation.includes('MODERATE') || interpretation.includes('PARTIAL')) return '#d97706';
    return '#dc2626';
  };

  // Helper: toggle item in array
  const toggleInArray = (arr, item) =>
    arr.includes(item) ? arr.filter(x => x !== item) : [...arr, item];

  // ---------- Synonym Results View ----------
  const renderSynonymResults = () => {
    if (!synonymResults) return null;
    return (
      <View style={{ marginTop: 15 }}>
        <View style={styles.resultsSummaryCard}>
          <Text style={styles.summaryTitle}>Overall Mean Jaccard</Text>
          <Text style={styles.summaryValue}>{synonymResults.overall_mean_jaccard?.toFixed(4)}</Text>
        </View>
        {synonymResults.clusters?.map((cluster, ci) => (
          <View key={ci} style={styles.clusterCard}>
            <View style={styles.clusterHeader}>
              <Text style={styles.clusterName}>{cluster.cluster}</Text>
              <View style={[styles.signalBadge, { backgroundColor: signalColor(cluster.interpretation) }]}>  
                <Text style={styles.signalBadgeText}>{cluster.interpretation}</Text>
              </View>
            </View>
            <Text style={styles.clusterWords}>Words: {cluster.words?.join(', ')}</Text>
            <View style={styles.metricsRow}>
              <View style={styles.metricBox}>
                <Text style={styles.metricLabel}>Mean Jaccard</Text>
                <Text style={styles.metricValue}>{cluster.mean_jaccard?.toFixed(4)}</Text>
              </View>
              <View style={styles.metricBox}>
                <Text style={styles.metricLabel}>Mean Cosine</Text>
                <Text style={styles.metricValue}>{cluster.mean_cosine_sim?.toFixed(4)}</Text>
              </View>
              <View style={styles.metricBox}>
                <Text style={styles.metricLabel}>All-shared</Text>
                <Text style={styles.metricValue}>{cluster.universal_shared_features?.length ?? 0}</Text>
              </View>
            </View>

            {/* Universal Shared Features */}
            {cluster.universal_shared_features?.length > 0 && (
              <View style={{ marginBottom: 15 }}>
                <Text style={styles.pairwiseTitle}>Universal Shared Features (all words)</Text>
                <View style={styles.featureChipList}>
                  {cluster.universal_shared_features.map(feat => (
                    <TouchableOpacity 
                      key={feat.id} 
                      style={styles.smallFeatureChip}
                      onPress={() => setSelectedFeature(feat)}
                    >
                      <Text style={styles.featureChipText}>{feat.label || `Feature ${feat.id}`}</Text>
                      <Text style={styles.featureChipId}>#{feat.id}</Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>
            )}

            {/* Pairwise detail */}
            <Text style={styles.pairwiseTitle}>Pairwise Comparison</Text>
            {cluster.pairwise?.map((pw, pi) => (
              <View key={pi} style={styles.pairCard}>
                <View style={styles.pairRow}>
                  <Text style={styles.pairWords}>{pw.word_a} ↔ {pw.word_b}</Text>
                  <Text style={styles.pairStat}>J={pw.jaccard?.toFixed(3)}</Text>
                  <Text style={styles.pairStat}>cos={pw.cosine_sim?.toFixed(3)}</Text>
                  <Text style={styles.pairStat}>shared={pw.shared_feature_count}</Text>
                </View>
                {pw.shared_features?.length > 0 && (
                  <View style={styles.featureChipList}>
                    {pw.shared_features.map(feat => (
                      <TouchableOpacity 
                        key={feat.id} 
                        style={styles.smallFeatureChip}
                        onPress={() => setSelectedFeature(feat)}
                      >
                        <Text style={styles.featureChipText}>{feat.label || `Feature ${feat.id}`}</Text>
                        <Text style={styles.featureChipId}>#{feat.id}</Text>
                      </TouchableOpacity>
                    ))}
                  </View>
                )}
              </View>
            ))}
          </View>
        ))}
      </View>
    );
  };

  // ---------- Caps Results View ----------
  const renderCapsResults = () => {
    if (!capsResults) return null;
    return (
      <View style={{ marginTop: 15 }}>
        <View style={styles.resultsSummaryCard}>
          <View style={styles.metricsRow}>
            <View style={styles.metricBox}>
              <Text style={styles.summaryTitle}>Overall Mean Jaccard</Text>
              <Text style={styles.summaryValue}>{capsResults.overall_mean_jaccard?.toFixed(4)}</Text>
            </View>
            <View style={styles.metricBox}>
              <Text style={styles.summaryTitle}>Overall Mean Cosine</Text>
              <Text style={styles.summaryValue}>{capsResults.overall_mean_cosine?.toFixed(4)}</Text>
            </View>
          </View>
        </View>
        {capsResults.words?.map((wordResult, wi) => (
          <View key={wi} style={styles.clusterCard}>
            <View style={styles.clusterHeader}>
              <Text style={styles.clusterName}>"{wordResult.word}"</Text>
              <View style={[styles.signalBadge, { backgroundColor: signalColor(wordResult.interpretation) }]}>
                <Text style={styles.signalBadgeText}>{wordResult.interpretation}</Text>
              </View>
            </View>
            <Text style={styles.clusterWords}>
              Variants: {wordResult.variants?.map(v => v.form).join(', ')}
            </Text>
            <View style={styles.metricsRow}>
              <View style={styles.metricBox}>
                <Text style={styles.metricLabel}>Mean Jaccard</Text>
                <Text style={styles.metricValue}>{wordResult.mean_jaccard?.toFixed(4)}</Text>
              </View>
              <View style={styles.metricBox}>
                <Text style={styles.metricLabel}>Mean Cosine</Text>
                <Text style={styles.metricValue}>{wordResult.mean_cosine_sim?.toFixed(4)}</Text>
              </View>
              {wordResult.lower_vs_upper && (
                <View style={styles.metricBox}>
                  <Text style={styles.metricLabel}>lower↔UPPER Jaccard</Text>
                  <Text style={styles.metricValue}>{wordResult.lower_vs_upper.jaccard?.toFixed(4)}</Text>
                </View>
              )}
              <View style={styles.metricBox}>
                <Text style={styles.metricLabel}>All-shared</Text>
                <Text style={styles.metricValue}>{wordResult.universal_shared_features?.length ?? 0}</Text>
              </View>
            </View>
            <Text style={styles.pairwiseTitle}>Pairwise Comparison</Text>
            {wordResult.pairwise?.map((pw, pi) => (
              <View key={pi} style={styles.pairRow}>
                <Text style={styles.pairWords}>{pw.variant_a} ↔ {pw.variant_b}</Text>
                <Text style={styles.pairStat}>J={pw.jaccard?.toFixed(3)}</Text>
                <Text style={styles.pairStat}>cos={pw.cosine_sim?.toFixed(3)}</Text>
                <Text style={styles.pairStat}>shared={pw.shared_feature_count}</Text>
              </View>
            ))}
          </View>
        ))}
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <ScrollView style={styles.scrollContainer}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Feature Interpretation using SAE by ANTLR</Text>
      </View>

      {/* Tab Bar */}
      <View style={styles.tabBar}>
        <TouchableOpacity
          style={[styles.tabItem, activeTab === TAB_SAE && styles.tabItemActive]}
          onPress={() => setActiveTab(TAB_SAE)}
        >
          <Text style={[styles.tabText, activeTab === TAB_SAE && styles.tabTextActive]}>SAE Analysis</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tabItem, activeTab === TAB_SYNONYM && styles.tabItemActive]}
          onPress={() => setActiveTab(TAB_SYNONYM)}
        >
          <Text style={[styles.tabText, activeTab === TAB_SYNONYM && styles.tabTextActive]}>Context-Similarity Test</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tabItem, activeTab === TAB_CAPS && styles.tabItemActive]}
          onPress={() => setActiveTab(TAB_CAPS)}
        >
          <Text style={[styles.tabText, activeTab === TAB_CAPS && styles.tabTextActive]}>Caps Test</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tabItem, activeTab === TAB_FEATURE_MAP && styles.tabItemActive]}
          onPress={() => setActiveTab(TAB_FEATURE_MAP)}
        >
          <Text style={[styles.tabText, activeTab === TAB_FEATURE_MAP && styles.tabTextActive]}>Feature Map</Text>
        </TouchableOpacity>
      </View>

      {/* ===================== SAE TAB ===================== */}
      {activeTab === TAB_SAE && (
      <View style={[styles.contentContainer, isLargeScreen ? styles.row : styles.column]}>
        {/* Left Panel: Input and Interactive Text */}
        <View style={[styles.panel, styles.leftPanel, isLargeScreen && styles.leftPanelLarge]}>
          <View style={styles.layerSelectorContainer}>
            <Text style={styles.layerSelectorLabel}>SAE Checkpoint</Text>
            <View style={styles.layerPickerWrapper}>
              {modelsLoading ? (
                <ActivityIndicator size="small" style={{ padding: 10 }} />
              ) : (
                <Picker
                  selectedValue={selectedModel}
                  onValueChange={(value) => setSelectedModel(value)}
                  style={styles.layerPicker}
                >
                  {models.map((m) => (
                    <Picker.Item
                      key={m.id}
                      label={`${m.name}  (${m.d_hidden || '?'} features)`}
                      value={m.id}
                    />
                  ))}
                </Picker>
              )}
            </View>
          </View>

          <Text style={styles.sectionTitle}>Input Text</Text>
          <View style={styles.inputCard}>
            <TextInput
              style={styles.textInput}
              multiline
              value={inputText}
              onChangeText={setInputText}
              placeholder="Type text to analyze..."
            />
          </View>
          <TouchableOpacity 
            style={styles.button} 
            onPress={() => analyzeText(inputText)}
            disabled={loading}
          >
            <Text style={styles.buttonText}>{loading ? "ANALYZING..." : "ANALYZE"}</Text>
          </TouchableOpacity>

          <Text style={styles.instructionText}>Click on any token to see its activated SAE features.</Text>
          
          <View style={styles.tokenContainer}>
             <Text style={styles.tokenWrapper}>
              {tokens.map((token, index) => (
                <Text
                  key={index}
                  style={[
                    styles.tokenText,
                    (activeTokenIndex === index || hoveredTokenIndex === index) && styles.tokenActive
                  ]}
                  onPress={() => handleTokenClick(index)}
                  {...(Platform.OS === 'web' ? {
                    onMouseEnter: () => {
                      if (activeTokenIndex === null) {
                        setHoveredTokenIndex(index);
                      }
                    },
                    onMouseLeave: () => setHoveredTokenIndex(null)
                  } : {})}
                >
                  {token.text}
                </Text>
              ))}
            </Text>
          </View>
        </View>

        {/* Right Panel: Activated Feature */}
        <View style={[styles.panel, styles.rightPanel, isLargeScreen && styles.rightPanelLarge]}>
          <View style={styles.featuresHeaderRow}>
             <Text style={styles.sectionTitle}>Activated Features</Text>
             
             {/* K Selector */}
             <View style={styles.kSelector}>
               <Text style={styles.kLabel}>Top K:</Text>
               <TextInput 
                 style={styles.kInput}
                 value={String(topK)}
                 onChangeText={(text) => {
                   const digitsOnly = text.replace(/[^0-9]/g, '');
                   if (digitsOnly === '') {
                     setTopK(0);
                   } else {
                     setTopK(Number(digitsOnly));
                   }
                 }}
                 keyboardType="numeric"
                 maxLength={4}
               />
             </View>
          </View>
          <Text style={styles.kHint}>Set Top K to 0 to show all active features and scroll down the page.</Text>
          
          {activeToken ? (
             <View>
                 <View style={styles.activeTokenChip}>
                   <Text style={styles.activeTokenText}>{activeToken.text.trim()}</Text>
                   <TouchableOpacity onPress={() => setActiveTokenIndex(null)}>
                      <Text style={styles.closeButton}>✕ SHOW ALL</Text>
                   </TouchableOpacity>
                 </View>

                 <Text style={styles.featureCountText}>
                   Showing {visibleActiveFeatures.length} of {activeToken.features.length} active features
                 </Text>

                 <View>
                 {visibleActiveFeatures.map((feature) => (
                    <View key={feature.id} style={styles.featureCard}>
                        <View style={styles.featureHeader}>
                        <View style={styles.dot} />
                        <Text style={styles.featureDescription}>{feature.description}</Text>
                        
                        <TouchableOpacity style={styles.idBox} onPress={() => setSelectedFeature(feature)}>
                            <Text style={styles.idText}>ID {feature.id} ↗</Text>
                        </TouchableOpacity>
                        </View>
                        <View style={styles.featureStats}>
                        <Text style={styles.tokenCount}>{feature.activation?.toFixed(2) ?? feature.tokens}</Text>
                        <Text style={styles.tokenLabel}>ACTIVATION</Text>
                        </View>
                    </View>
                 ))}
                     </View>
                 {activeToken.features.length === 0 && (
                     <Text>No active features for this token.</Text>
                 )}
            </View>
          ) : (
             <View style={styles.placeholderCard}>
               <Text style={styles.placeholderText}>
                 Click on a word on the left to inspect its features!
               </Text>
               <Text style={styles.placeholderSubText}>
                 Each token shows which SAE features activate most strongly.
               </Text>
             </View>
          )}

          {error && (
             <View style={styles.errorBox}>
               <Text style={styles.errorText}>Backend Error: {error}</Text>
             </View>
          )}
        </View>
      </View>
      )}

      {/* ===================== SYNONYM TAB ===================== */}
      {activeTab === TAB_SYNONYM && (
      <View style={styles.contentContainer}>
        <View style={styles.panel}>
          {/* Model selector (shared) */}
          <View style={styles.layerSelectorContainer}>
            <Text style={styles.layerSelectorLabel}>SAE Checkpoint</Text>
            <View style={styles.layerPickerWrapper}>
              {modelsLoading ? (
                <ActivityIndicator size="small" style={{ padding: 10 }} />
              ) : (
                <Picker
                  selectedValue={selectedModel}
                  onValueChange={(value) => setSelectedModel(value)}
                  style={styles.layerPicker}
                >
                  {models.map((m) => (
                    <Picker.Item key={m.id} label={`${m.name}  (${m.d_hidden || '?'} features)`} value={m.id} />
                  ))}
                </Picker>
              )}
            </View>
          </View>

          <Text style={styles.sectionTitle}>Context-Similarity Test</Text>
          <Text style={styles.instructionText}>
            Tests whether SAE features respond to meaning rather than surface form by
            comparing feature overlap between words you believe are related.
          </Text>

          {/* Mode toggle */}
          <View style={styles.modeToggleRow}>
            <TouchableOpacity
              style={[styles.modeButton, synonymMode === 'custom' && styles.modeButtonActive]}
              onPress={() => setSynonymMode('custom')}
            >
              <Text style={[styles.modeButtonText, synonymMode === 'custom' && styles.modeButtonTextActive]}>Custom Words</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.modeButton, synonymMode === 'preset' && styles.modeButtonActive]}
              onPress={() => setSynonymMode('preset')}
            >
              <Text style={[styles.modeButtonText, synonymMode === 'preset' && styles.modeButtonTextActive]}>Preset Clusters</Text>
            </TouchableOpacity>
          </View>

          {synonymMode === 'custom' ? (
            <View>
              <Text style={styles.inputLabel}>Enter words to compare (comma-separated, min 2):</Text>
              <View style={styles.inputCard}>
                <TextInput
                  style={styles.textInput}
                  value={customSynonymWords}
                  onChangeText={setCustomSynonymWords}
                  placeholder="e.g. happy, joyful, elated, cheerful"
                  multiline={false}
                />
              </View>
              <Text style={styles.hintText}>
                Sentences are generated automatically for each word. The model extracts features
                at each word's position and compares overlap via Jaccard similarity.
              </Text>
            </View>
          ) : (
            <View>
              <Text style={styles.inputLabel}>Select predefined clusters:</Text>
              <View style={styles.chipContainer}>
                {synonymClusters.map((c) => (
                  <TouchableOpacity
                    key={c}
                    style={[styles.chip, selectedSynonymClusters.includes(c) && styles.chipSelected]}
                    onPress={() => setSelectedSynonymClusters(toggleInArray(selectedSynonymClusters, c))}
                  >
                    <Text style={[styles.chipText, selectedSynonymClusters.includes(c) && styles.chipTextSelected]}>
                      {c} {synonymClusterDetails[c] ? `(${synonymClusterDetails[c].join(', ')})` : ''}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
          )}

          <View style={styles.featuresHeaderRow}>
            <TouchableOpacity
              style={[styles.button, synonymLoading && { opacity: 0.6 }]}
              onPress={runSynonymTest}
              disabled={synonymLoading}
            >
              <Text style={styles.buttonText}>
                {synonymLoading ? 'RUNNING...' : 'RUN CONTEXT-SIMILARITY TEST'}
              </Text>
            </TouchableOpacity>
            <View style={styles.kSelector}>
              <Text style={styles.kLabel}>Top K:</Text>
              <TextInput
                style={styles.kInput}
                value={String(synonymTopK)}
                onChangeText={(t) => setSynonymTopK(Number(t.replace(/[^0-9]/g, '')) || 1)}
                keyboardType="numeric"
                maxLength={3}
              />
            </View>
          </View>

          {synonymLoading && <ActivityIndicator size="large" style={{ marginTop: 20 }} />}

          {renderSynonymResults()}

          {error && !synonymLoading && (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>Error: {error}</Text>
            </View>
          )}
        </View>
      </View>
      )}

      {/* ===================== CAPS TAB ===================== */}
      {activeTab === TAB_CAPS && (
      <View style={styles.contentContainer}>
        <View style={styles.panel}>
          <View style={styles.layerSelectorContainer}>
            <Text style={styles.layerSelectorLabel}>SAE Checkpoint</Text>
            <View style={styles.layerPickerWrapper}>
              {modelsLoading ? (
                <ActivityIndicator size="small" style={{ padding: 10 }} />
              ) : (
                <Picker
                  selectedValue={selectedModel}
                  onValueChange={(value) => setSelectedModel(value)}
                  style={styles.layerPicker}
                >
                  {models.map((m) => (
                    <Picker.Item key={m.id} label={`${m.name}  (${m.d_hidden || '?'} features)`} value={m.id} />
                  ))}
                </Picker>
              )}
            </View>
          </View>

          <Text style={styles.sectionTitle}>Capitalisation Invariance Test</Text>
          <Text style={styles.instructionText}>
            Tests whether SAE features are stable across capitalisation variants (e.g. cat / Cat / CAT).
            Select words below or leave empty to test all.
          </Text>

          {/* Word chips */}
          <View style={styles.chipContainer}>
            {capsWords.map((w) => (
              <TouchableOpacity
                key={w}
                style={[styles.chip, selectedCapsWords.includes(w) && styles.chipSelected]}
                onPress={() => setSelectedCapsWords(toggleInArray(selectedCapsWords, w))}
              >
                <Text style={[styles.chipText, selectedCapsWords.includes(w) && styles.chipTextSelected]}>{w}</Text>
              </TouchableOpacity>
            ))}
          </View>

          <TouchableOpacity
            style={[styles.button, capsLoading && { opacity: 0.6 }]}
            onPress={runCapsTest}
            disabled={capsLoading}
          >
            <Text style={styles.buttonText}>
              {capsLoading ? 'RUNNING CAPS TEST...' : 'RUN CAPS TEST'}
            </Text>
          </TouchableOpacity>

          {capsLoading && <ActivityIndicator size="large" style={{ marginTop: 20 }} />}

          {renderCapsResults()}

          {error && !capsLoading && (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>Error: {error}</Text>
            </View>
          )}
        </View>
      </View>
      )}

      {/* ===================== FEATURE MAP TAB ===================== */}
      {activeTab === TAB_FEATURE_MAP && (
      <View style={styles.contentContainer}>
        <View style={styles.panel}>
          <Text style={styles.sectionTitle}>Feature Map (Human-Readable)</Text>
          <Text style={styles.instructionText}>
            Maps activated tokens to their features with local sentence context.
            Use this view to understand what each feature is responding to.
          </Text>

          <View style={styles.inputCard}>
            <Text style={styles.featureMapSentenceLabel}>Current Sentence</Text>
            <Text style={styles.featureMapSentence}>{inputText}</Text>
          </View>

          <View style={styles.featureMapMetaRow}>
            <Text style={styles.featureMapMetaText}>Activated tokens: {activatedTokenRows.length}</Text>
            <Text style={styles.featureMapMetaText}>Top K per token: {topK === 0 ? 'All' : topK}</Text>
          </View>

          <Text style={styles.subSectionTitle}>Feature -> All Dataset Activations</Text>
          <View style={styles.featureLookupCard}>
            <Text style={styles.instructionText}>
              Select any feature ID and fetch sentences (+tokens) from MLCommons/peoples_speech where it activates.
            </Text>

            <View style={styles.featureLookupRow}>
              <View style={styles.featureLookupField}>
                <Text style={styles.inputLabel}>Feature ID</Text>
                <TextInput
                  style={styles.featureLookupInput}
                  value={lookupFeatureId}
                  onChangeText={(t) => setLookupFeatureId(t.replace(/[^0-9]/g, ''))}
                  keyboardType="numeric"
                  placeholder="e.g. 42"
                />
              </View>
              <View style={styles.featureLookupField}>
                <Text style={styles.inputLabel}>Max Sentences</Text>
                <TextInput
                  style={styles.featureLookupInput}
                  value={lookupMaxSentences}
                  onChangeText={(t) => setLookupMaxSentences(t.replace(/[^0-9]/g, ''))}
                  keyboardType="numeric"
                  placeholder="200"
                />
              </View>
              <View style={styles.featureLookupField}>
                <Text style={styles.inputLabel}>Max Results</Text>
                <TextInput
                  style={styles.featureLookupInput}
                  value={lookupMaxResults}
                  onChangeText={(t) => setLookupMaxResults(t.replace(/[^0-9]/g, ''))}
                  keyboardType="numeric"
                  placeholder="100"
                />
              </View>
              <View style={styles.featureLookupField}>
                <Text style={styles.inputLabel}>Min Activation</Text>
                <TextInput
                  style={styles.featureLookupInput}
                  value={lookupMinActivation}
                  onChangeText={(t) => setLookupMinActivation(t.replace(/[^0-9.]/g, ''))}
                  keyboardType="numeric"
                  placeholder="0.0"
                />
              </View>
            </View>

            <View style={styles.featureLookupActionsRow}>
              <TouchableOpacity
                style={[styles.button, featureLookupLoading && { opacity: 0.6 }]}
                onPress={runFeatureLookup}
                disabled={featureLookupLoading}
              >
                <Text style={styles.buttonText}>{featureLookupLoading ? 'SEARCHING...' : 'FIND ACTIVATIONS'}</Text>
              </TouchableOpacity>
              {!!selectedFeature?.id && (
                <TouchableOpacity
                  style={styles.secondaryButton}
                  onPress={() => setLookupFeatureId(String(selectedFeature.id))}
                >
                  <Text style={styles.secondaryButtonText}>Use Selected Feature #{selectedFeature.id}</Text>
                </TouchableOpacity>
              )}
            </View>

            {featureLookupError && (
              <View style={styles.errorBox}>
                <Text style={styles.errorText}>{featureLookupError}</Text>
              </View>
            )}

            {featureLookupResults && (
              <View style={styles.featureLookupResultsWrap}>
                <View style={styles.resultsSummaryCard}>
                  <Text style={styles.summaryTitle}>Feature</Text>
                  <Text style={styles.summaryValue}>#{featureLookupResults.feature_id}</Text>
                  <Text style={styles.clusterWords}>{featureLookupResults.feature_description}</Text>
                  <Text style={styles.clusterWords}>
                    Dataset: {featureLookupResults.dataset} [{featureLookupResults.split}] | Scanned {featureLookupResults.scanned_sentences} sentences / {featureLookupResults.scanned_tokens} tokens
                  </Text>
                  <Text style={styles.clusterWords}>
                    Matches: {featureLookupResults.matches?.length ?? 0} shown ({featureLookupResults.total_matches ?? 0} total)
                  </Text>
                </View>

                {(featureLookupResults.matches || []).map((match, idx) => (
                  <View key={`lookup-match-${idx}`} style={styles.featureLookupMatchCard}>
                    <View style={styles.featureMapTokenHeader}>
                      <Text style={styles.featureMapTokenIndex}>Sentence #{match.sentence_index} | Token #{match.token_index}</Text>
                      <Text style={styles.featureMapTokenCount}>Activation {match.activation?.toFixed?.(4) ?? match.activation}</Text>
                    </View>
                    <Text style={styles.featureMapContextText}>
                      {match.left_context}
                      <Text style={styles.featureMapContextToken}>{match.token}</Text>
                      {match.right_context}
                    </Text>
                    <Text style={styles.featureLookupSentence}>{match.sentence}</Text>
                  </View>
                ))}
              </View>
            )}
          </View>

          {tokens.length === 0 && (
            <View style={styles.placeholderCard}>
              <Text style={styles.placeholderText}>No analysis found yet.</Text>
              <Text style={styles.placeholderSubText}>Run ANALYZE in SAE Analysis tab first.</Text>
            </View>
          )}

          {tokens.length > 0 && activatedTokenRows.length === 0 && (
            <View style={styles.placeholderCard}>
              <Text style={styles.placeholderText}>No activated tokens in current result.</Text>
              <Text style={styles.placeholderSubText}>Try a different sentence or model checkpoint.</Text>
            </View>
          )}

          {activatedTokenRows.length > 0 && (
            <View>
              <Text style={styles.subSectionTitle}>Token -> Features</Text>
              {activatedTokenRows.map((row) => (
                <View key={`token-${row.tokenIndex}`} style={styles.featureMapTokenCard}>
                  <View style={styles.featureMapTokenHeader}>
                    <Text style={styles.featureMapTokenIndex}>Token #{row.tokenIndex}</Text>
                    <Text style={styles.featureMapTokenCount}>Features {row.features.length}/{row.totalFeatures}</Text>
                  </View>
                  <Text style={styles.featureMapContextText}>
                    {row.before}
                    <Text style={styles.featureMapContextToken}>{row.tokenText}</Text>
                    {row.after}
                  </Text>
                  {row.features.map((feature) => (
                    <TouchableOpacity
                      key={`token-${row.tokenIndex}-feature-${feature.id}`}
                      style={styles.featureMapFeatureRow}
                      onPress={() => setSelectedFeature(feature)}
                    >
                      <Text style={styles.featureMapFeatureId}>#{feature.id}</Text>
                      <Text style={styles.featureMapFeatureDesc}>{feature.description}</Text>
                      <Text style={styles.featureMapFeatureActivation}>
                        {feature.activation?.toFixed(2) ?? 'n/a'}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              ))}

              <Text style={styles.subSectionTitle}>Feature -> Activated Tokens</Text>
              {featureToTokensRows.map((featureRow) => (
                <View key={`feature-${featureRow.id}`} style={styles.featureMapFeatureCard}>
                  <Text style={styles.featureMapFeatureCardTitle}>#{featureRow.id} {featureRow.description}</Text>
                  <Text style={styles.featureMapFeatureCardMeta}>Activated by {featureRow.tokens.length} token(s)</Text>
                  <View style={styles.featureMapTokenChips}>
                    {featureRow.tokens.map((entry, idx) => (
                      <View key={`feature-${featureRow.id}-token-${idx}`} style={styles.featureMapTokenChip}>
                        <Text style={styles.featureMapTokenChipText}>
                          {entry.tokenText.trim() || '(space)'} @ {entry.tokenIndex}
                        </Text>
                      </View>
                    ))}
                  </View>
                </View>
              ))}
            </View>
          )}
        </View>
      </View>
      )}

      </ScrollView>
      
      <View style={styles.footer}>
        <Text style={styles.footerText}>Created by: Nafis Nahian, Arnob Biswas, Tanvir Liaquat Uday</Text>
      </View>
      
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
        />
      </Modal>

      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  scrollContainer: {
    flex: 1,
  },
  footer: {
    padding: 20,
    backgroundColor: '#f8f9fa',
    borderTopWidth: 1,
    borderTopColor: '#eee',
    alignItems: 'center',
    marginTop: 20,
  },
  footerText: {
    fontSize: 12,
    color: '#666',
    marginBottom: 5,
  },
  footerCredits: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  header: {
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  contentContainer: {
    padding: 20,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  column: {
    flexDirection: 'column',
  },
  panel: {
    padding: 10,
    marginBottom: 20,
  },
  leftPanel: {
    flex: 1, // Takes up more space on large screens usually
  },
  leftPanelLarge: {
    flex: 6, // 60%
    marginRight: 20,
  },
  rightPanel: {
    flex: 1,
  },
  rightPanelLarge: {
    flex: 4, // 40%
    marginLeft: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  layerSelectorContainer: {
    marginBottom: 14,
  },
  layerSelectorLabel: {
    fontSize: 14,
    color: '#555',
    marginBottom: 6,
    fontWeight: '600',
  },
  layerPickerWrapper: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    backgroundColor: '#fff',
    overflow: 'hidden',
  },
  layerPicker: {
    height: 44,
    width: '100%',
  },
  instructionText: {
    fontSize: 12,
    color: '#666',
    marginBottom: 10,
    fontStyle: 'italic',
  },
  inputCard: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 10,
    marginBottom: 15,
  },
  textInput: {
    minHeight: 80,
    fontSize: 16,
    color: '#333',
    textAlignVertical: 'top',
    outlineStyle: 'none',
  },
  button: {
    backgroundColor: '#4285F4',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 4,
    alignSelf: 'flex-start',
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14,
  },
  tokenContainer: {
    backgroundColor: '#fff', // White background as per screenshot
    padding: 20,
    borderRadius: 8,
    minHeight: 100,
  },
  tokenWrapper: {
    flexDirection: 'row', 
    flexWrap: 'wrap',
  },
  tokenText: {
    fontSize: 18,
    lineHeight: 32,
    color: '#444',
    paddingHorizontal: 2,
    paddingVertical: 2,
    borderRadius: 4,
  },
  tokenActive: {
    backgroundColor: '#00897b', // Dark green selection
    color: '#fff', // White text on selection
    fontWeight: 'bold',
  },
  
  // Right Panel specific
  featuresHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  kSelector: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  kLabel: {
    fontSize: 14,
    marginRight: 5,
    color: '#555',
  },
  kInput: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 4,
    padding: 5,
    width: 56,
    textAlign: 'center',
  },
  kHint: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 10,
  },

  activeTokenChip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#a7f3d0', // Light green
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    alignSelf: 'flex-start',
    marginBottom: 20,
  },
  activeTokenText: {
    fontWeight: 'bold',
    color: '#064e3b',
    marginRight: 10,
  },
  closeButton: {
    fontSize: 10,
    color: '#ef4444', // Red-ish for close/clear
    fontWeight: 'bold',
  },

  featureCountText: {
    fontSize: 12,
    color: '#4b5563',
    marginBottom: 10,
  },

  // Feature Card
  featureCard: {
    backgroundColor: '#e6fcf5',
    borderRadius: 12,
    padding: 16,
    marginBottom: 10,
  },
  featureHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    marginBottom: 5,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#00897b',
    marginTop: 6,
    marginRight: 10,
  },
  featureDescription: {
    flex: 1,
    fontSize: 14,
    lineHeight: 20,
    color: '#004d40',
    marginRight: 10,
  },
  idBox: {
    // borderWidth: 1,
    // borderColor: '#00cc66',
    // borderRadius: 4,
    // padding: 2,
  },
  idText: {
    fontSize: 10,
    color: '#00897b',
    textDecorationLine: 'underline',
  },
  featureStats: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    marginTop: 5,
  },
  tokenCount: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#004d40',
    marginRight: 4,
  },
  tokenLabel: {
    fontSize: 8,
    color: '#004d40',
    fontWeight: 'bold',
  },
  
  placeholderCard: {
    padding: 30,
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    borderStyle: 'dashed',
    borderWidth: 2,
    borderColor: '#e5e7eb',
  },
  placeholderText: {
    textAlign: 'center',
    fontSize: 16,
    color: '#6b7280',
    marginBottom: 10,
  },
  placeholderSubText: {
    textAlign: 'center',
    fontSize: 14,
    color: '#9ca3af',
  },
  errorBox: {
    marginTop: 20,
    padding: 10,
    backgroundColor: '#ffebee',
    borderRadius: 4,
  },
  errorText: {
    color: '#c62828',
  },

  // ── Tab Bar ──────────────────────────────────────────────────────────────
  tabBar: {
    flexDirection: 'row',
    borderBottomWidth: 2,
    borderBottomColor: '#e5e7eb',
    backgroundColor: '#fff',
  },
  tabItem: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
    borderBottomWidth: 3,
    borderBottomColor: 'transparent',
  },
  tabItemActive: {
    borderBottomColor: '#4285F4',
  },
  tabText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#6b7280',
  },
  tabTextActive: {
    color: '#4285F4',
  },

  // ── Chip selector ────────────────────────────────────────────────────────
  chipContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 15,
    gap: 8,
  },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#d1d5db',
    backgroundColor: '#f9fafb',
  },
  chipSelected: {
    backgroundColor: '#4285F4',
    borderColor: '#4285F4',
  },
  chipText: {
    fontSize: 13,
    color: '#374151',
    fontWeight: '500',
  },
  chipTextSelected: {
    color: '#fff',
  },

  // ── Test result cards ────────────────────────────────────────────────────
  resultsSummaryCard: {
    backgroundColor: '#f0f9ff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: '#bfdbfe',
  },
  summaryTitle: {
    fontSize: 13,
    color: '#64748b',
    fontWeight: '600',
    marginBottom: 4,
  },
  summaryValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1e40af',
  },
  clusterCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  clusterHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  clusterName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    textTransform: 'capitalize',
  },
  signalBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  signalBadgeText: {
    fontSize: 11,
    fontWeight: 'bold',
    color: '#fff',
  },
  clusterWords: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 10,
  },
  metricsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 12,
  },
  metricBox: {
    backgroundColor: '#f8fafc',
    borderRadius: 8,
    padding: 10,
    minWidth: 120,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  metricLabel: {
    fontSize: 11,
    color: '#64748b',
    fontWeight: '600',
    marginBottom: 2,
  },
  metricValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#0f172a',
  },
  pairwiseTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginTop: 6,
    marginBottom: 6,
  },
  pairRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
    gap: 12,
  },
  pairWords: {
    fontSize: 13,
    fontWeight: '600',
    color: '#334155',
    minWidth: 160,
  },
  pairStat: {
    fontSize: 12,
    color: '#64748b',
    fontFamily: Platform.OS === 'web' ? 'monospace' : undefined,
  },
  pairCard: {
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
    paddingBottom: 8,
    marginBottom: 4,
  },
  featureChipList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    paddingHorizontal: 8,
    marginTop: 4,
  },
  smallFeatureChip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f1f5f9',
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  featureChipText: {
    fontSize: 11,
    color: '#334155',
    fontWeight: '500',
    marginRight: 4,
  },
  featureChipId: {
    fontSize: 10,
    color: '#64748b',
    fontWeight: 'bold',
  },

  // ── Mode toggle ──────────────────────────────────────────────────────────
  modeToggleRow: {
    flexDirection: 'row',
    marginBottom: 15,
    gap: 8,
  },
  modeButton: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#d1d5db',
    backgroundColor: '#f9fafb',
  },
  modeButtonActive: {
    backgroundColor: '#4285F4',
    borderColor: '#4285F4',
  },
  modeButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
  },
  modeButtonTextActive: {
    color: '#fff',
  },

  // ── Input helpers ────────────────────────────────────────────────────────
  inputLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 6,
  },
  hintText: {
    fontSize: 12,
    color: '#9ca3af',
    fontStyle: 'italic',
    marginBottom: 14,
    lineHeight: 18,
  },

  // Feature map tab
  subSectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937',
    marginTop: 10,
    marginBottom: 10,
  },
  featureMapSentenceLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 6,
  },
  featureMapSentence: {
    fontSize: 14,
    color: '#111827',
    lineHeight: 22,
  },
  featureMapMetaRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 12,
  },
  featureMapMetaText: {
    fontSize: 12,
    color: '#4b5563',
    fontWeight: '600',
  },
  featureMapTokenCard: {
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 10,
    padding: 12,
    marginBottom: 10,
  },
  featureMapTokenHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  featureMapTokenIndex: {
    fontSize: 12,
    color: '#374151',
    fontWeight: '700',
  },
  featureMapTokenCount: {
    fontSize: 12,
    color: '#6b7280',
    fontWeight: '600',
  },
  featureMapContextText: {
    fontSize: 14,
    color: '#374151',
    lineHeight: 22,
    marginBottom: 8,
  },
  featureMapContextToken: {
    backgroundColor: '#d1fae5',
    color: '#065f46',
    fontWeight: '700',
    borderRadius: 4,
  },
  featureMapFeatureRow: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ecfeff',
    backgroundColor: '#f0fdfa',
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 10,
    marginBottom: 6,
    gap: 8,
  },
  featureMapFeatureId: {
    fontSize: 11,
    fontWeight: '700',
    color: '#0f766e',
    minWidth: 42,
  },
  featureMapFeatureDesc: {
    flex: 1,
    fontSize: 13,
    color: '#134e4a',
  },
  featureMapFeatureActivation: {
    fontSize: 12,
    fontWeight: '700',
    color: '#0f766e',
    minWidth: 36,
    textAlign: 'right',
  },
  featureMapFeatureCard: {
    backgroundColor: '#f8fafc',
    borderWidth: 1,
    borderColor: '#e2e8f0',
    borderRadius: 10,
    padding: 12,
    marginBottom: 10,
  },
  featureMapFeatureCardTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 4,
  },
  featureMapFeatureCardMeta: {
    fontSize: 12,
    color: '#64748b',
    marginBottom: 8,
  },
  featureMapTokenChips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  featureMapTokenChip: {
    backgroundColor: '#e2e8f0',
    borderRadius: 14,
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  featureMapTokenChipText: {
    fontSize: 12,
    color: '#334155',
    fontWeight: '600',
  },
  featureLookupCard: {
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#dbeafe',
    borderRadius: 10,
    padding: 12,
    marginBottom: 14,
  },
  featureLookupRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  featureLookupField: {
    minWidth: 120,
    flexGrow: 1,
  },
  featureLookupInput: {
    borderWidth: 1,
    borderColor: '#cbd5e1',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 8,
    fontSize: 14,
    color: '#111827',
    outlineStyle: 'none',
  },
  featureLookupActionsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    alignItems: 'center',
    marginTop: 12,
    marginBottom: 6,
  },
  secondaryButton: {
    borderWidth: 1,
    borderColor: '#93c5fd',
    backgroundColor: '#eff6ff',
    borderRadius: 6,
    paddingVertical: 9,
    paddingHorizontal: 12,
  },
  secondaryButtonText: {
    color: '#1d4ed8',
    fontWeight: '700',
    fontSize: 12,
  },
  featureLookupResultsWrap: {
    marginTop: 12,
  },
  featureLookupMatchCard: {
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#e2e8f0',
    borderRadius: 10,
    padding: 12,
    marginBottom: 10,
  },
  featureLookupSentence: {
    fontSize: 13,
    color: '#475569',
    lineHeight: 20,
    marginTop: 4,
  },
});
