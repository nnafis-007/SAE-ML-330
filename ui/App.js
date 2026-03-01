import { StatusBar } from 'expo-status-bar';
import { useEffect, useState } from 'react';
import { StyleSheet, Text, View, ScrollView, useWindowDimensions, TouchableOpacity, ActivityIndicator, TextInput, Platform, Modal } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import FeatureDetails from './FeatureDetails';

const DEFAULT_TEXT = "Did you know that pineapples were a symbol of hospitality in colonial America? This exotic fruit, once a rare delicacy, was often displayed at gatherings to impress guests.";
const API_BASE = 'http://localhost:8000';

// Tab constants
const TAB_SAE = 'sae';
const TAB_SYNONYM = 'synonym';
const TAB_CAPS = 'caps';

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
  const [topK, setTopK] = useState(3); // Default to Top 3 features
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
          top_k: topK * 3, // Fetch more from backend, slice per token in UI
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
            {/* Pairwise detail */}
            <Text style={styles.pairwiseTitle}>Pairwise Comparison</Text>
            {cluster.pairwise?.map((pw, pi) => (
              <View key={pi} style={styles.pairRow}>
                <Text style={styles.pairWords}>{pw.word_a} ↔ {pw.word_b}</Text>
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
        <Text style={styles.headerTitle}>Sparse AutoEncoder by ANTLR</Text>
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
             <Text style={styles.sectionTitle}>Top Features Activated</Text>
             
             {/* K Selector */}
             <View style={styles.kSelector}>
               <Text style={styles.kLabel}>Top K:</Text>
               <TextInput 
                 style={styles.kInput}
                 value={String(topK)}
                 onChangeText={(text) => setTopK(Number(text.replace(/[^0-9]/g, '')) || 1)}
                 keyboardType="numeric"
                 maxLength={2}
               />
             </View>
          </View>
          
          {activeToken ? (
             <View>
                 <View style={styles.activeTokenChip}>
                   <Text style={styles.activeTokenText}>{activeToken.text.trim()}</Text>
                   <TouchableOpacity onPress={() => setActiveTokenIndex(null)}>
                      <Text style={styles.closeButton}>✕ SHOW ALL</Text>
                   </TouchableOpacity>
                 </View>

                 <ScrollView style={styles.featuresScroll} nestedScrollEnabled={true}>
                 {activeToken.features.slice(0, topK).map((feature, idx) => (
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
                 </ScrollView>
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

      </ScrollView>
      
      <View style={styles.footer}>
        <Text style={styles.footerText}>© 2026 Sparse AutoEncoder. All rights reserved.</Text>
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
    width: 40,
    textAlign: 'center',
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

  // Feature Card
  featuresScroll: {
    maxHeight: 400, // Show roughly 5 items (approx 80px each)
  },
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
});
