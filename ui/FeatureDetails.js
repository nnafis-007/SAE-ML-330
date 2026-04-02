import React, { useState } from 'react';
import {
  ActivityIndicator,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';

const API_BASE = 'http://localhost:8001';
const FEATURE_LOOKUP_DATASET = 'openwebtext';
const FEATURE_LOOKUP_SPLIT = 'train';

export default function FeatureDetails({ feature, onClose, modelId }) {
  const [maxSentences, setMaxSentences] = useState('200');
  const [minActivation, setMinActivation] = useState('0.1');
  const [maxResults, setMaxResults] = useState('100');
  const [llmExamples, setLlmExamples] = useState('25');

  const [llmLabel, setLlmLabel] = useState(null);
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmError, setLlmError] = useState(null);

  const [activationPayload, setActivationPayload] = useState(null);
  const [analysisMeta, setAnalysisMeta] = useState(null);
  const [activationQueryKey, setActivationQueryKey] = useState(null);

  const fetchLlmLabel = async () => {
    if (!feature?.id || !modelId) {
      return;
    }

    const sentenceLimit = Math.max(1, Number(maxSentences) || 200);
    const minAct = Math.max(0, Number(minActivation) || 0);
    const resultLimit = Math.max(1, Number(maxResults) || 100);
    const llmExampleLimit = Math.max(1, Number(llmExamples) || 25);
    const queryKey = JSON.stringify({
      modelId,
      featureId: feature.id,
      sentenceLimit,
      minAct,
      resultLimit,
      dataset: FEATURE_LOOKUP_DATASET,
      split: FEATURE_LOOKUP_SPLIT,
    });

    setLlmLoading(true);
    setLlmLabel(null);
    setLlmError(null);
    setAnalysisMeta(null);

    try {
      // Step 1: fetch activating contexts from dataset with user-provided filters.
      // Reuse last response if all query params are unchanged.
      let activationData = activationPayload;
      if (!activationData || activationQueryKey !== queryKey) {
        const activationsRes = await fetch(`${API_BASE}/feature-activations`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model_id: modelId,
            feature_id: feature.id,
            dataset_name: FEATURE_LOOKUP_DATASET,
            dataset_config: null,
            split: FEATURE_LOOKUP_SPLIT,
            max_sentences: sentenceLimit,
            max_results: resultLimit,
            min_activation: minAct,
          }),
        });

        if (!activationsRes.ok) {
          const err = await activationsRes.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${activationsRes.status}`);
        }

        activationData = await activationsRes.json();
        setActivationPayload(activationData);
        setActivationQueryKey(queryKey);
      }

      const matches = activationData.matches || [];
      const uniqueSentences = Array.from(
        new Set(matches.map((m) => (m.sentence || '').trim()).filter(Boolean))
      ).slice(0, llmExampleLimit);

      if (uniqueSentences.length === 0) {
        throw new Error('No activating contexts found with the selected thresholds.');
      }

      // Step 2: run LLM labeling using only those sampled sentences.
      const labelRes = await fetch(`${API_BASE}/label-feature`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelId,
          feature_idx: feature.id,
          analyzer: 'sae',
          corpus_texts: uniqueSentences,
        }),
      });

      if (!labelRes.ok) {
        const err = await labelRes.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${labelRes.status}`);
      }

      const labelData = await labelRes.json();
      setLlmLabel(labelData);
      setAnalysisMeta({
        sentenceLimit,
        minAct,
        resultLimit,
        llmExampleLimit,
        corpusSentencesUsed: uniqueSentences.length,
        totalMatches: activationData.total_matches || 0,
        activationCacheHit: activationData.cache_hit === true,
        labelCacheHit: labelData.cache_hit === true,
      });
    } catch (e) {
      setLlmError(e.message);
    } finally {
      setLlmLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.topBar}>
        <TouchableOpacity style={styles.doneButton} onPress={onClose}>
          <Text style={styles.doneButtonText}>Done</Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.contentScroll}>
        <View style={styles.mainContent}>
          <View style={styles.headerRow}>
            <Text style={styles.pageTitle}>{feature?.description || `Feature ${feature?.id ?? ''}`}</Text>
            <View style={styles.modelCard}>
              <Text style={styles.modelInfo}>Pretrained SAE</Text>
              <Text style={styles.modelDetail}>LLM ANALYSIS</Text>
              <Text style={styles.modelIndex}>FEATURE {feature?.id || '-'} | PRETRAINED</Text>
            </View>
          </View>

          <View style={styles.analysisPanel}>
            <Text style={styles.panelTitle}>Run Feature LLM Analysis</Text>
            <Text style={styles.panelHint}>
              Configure how many sentences to scan and the minimum token activation threshold,
              then analyze this feature.
            </Text>

            <View style={styles.inputRow}>
              <View style={styles.inputField}>
                <Text style={styles.inputLabel}>How many sentences to analyze</Text>
                <TextInput
                  style={styles.numericInput}
                  value={maxSentences}
                  onChangeText={(t) => setMaxSentences(t.replace(/[^0-9]/g, ''))}
                  keyboardType="numeric"
                  placeholder="200"
                />
              </View>
              <View style={styles.inputField}>
                <Text style={styles.inputLabel}>Minimum activation value</Text>
                <TextInput
                  style={styles.numericInput}
                  value={minActivation}
                  onChangeText={(t) => setMinActivation(t.replace(/[^0-9.]/g, ''))}
                  keyboardType="numeric"
                  placeholder="0.1"
                />
              </View>
              <View style={styles.inputField}>
                <Text style={styles.inputLabel}>Maximum matches to keep</Text>
                <TextInput
                  style={styles.numericInput}
                  value={maxResults}
                  onChangeText={(t) => setMaxResults(t.replace(/[^0-9]/g, ''))}
                  keyboardType="numeric"
                  placeholder="100"
                />
              </View>
              <View style={styles.inputField}>
                <Text style={styles.inputLabel}>Activating examples sent to LLM</Text>
                <TextInput
                  style={styles.numericInput}
                  value={llmExamples}
                  onChangeText={(t) => setLlmExamples(t.replace(/[^0-9]/g, ''))}
                  keyboardType="numeric"
                  placeholder="25"
                />
              </View>
            </View>

            <TouchableOpacity
              style={[styles.actionButton, styles.llmButton]}
              onPress={fetchLlmLabel}
              disabled={llmLoading}
            >
              <Text style={styles.actionButtonText}>
                {llmLoading ? 'Analyzing...' : `Analyze Feature #${feature?.id}`}
              </Text>
            </TouchableOpacity>

            {llmLoading && (
              <View style={styles.loadingRow}>
                <ActivityIndicator size="small" color="#2563eb" />
                <Text style={styles.loadingText}>Collecting activations and requesting LLM label...</Text>
              </View>
            )}
          </View>

          <View style={styles.llmSection}>
            {llmError && <Text style={styles.llmError}>{llmError}</Text>}

            {llmLabel && (
              <View style={styles.llmResult}>
                <Text style={styles.resultTitle}>LLM Label Result</Text>
                <View style={styles.llmLabelRow}>
                  <Text style={styles.llmLabelText}>{llmLabel.label || '(no label returned)'}</Text>
                  <View
                    style={[
                      styles.confidenceBadge,
                      llmLabel.confidence === 'high'
                        ? styles.confHigh
                        : llmLabel.confidence === 'medium'
                          ? styles.confMed
                          : styles.confLow,
                    ]}
                  >
                    <Text style={styles.confidenceText}>{llmLabel.confidence?.toUpperCase()}</Text>
                  </View>
                </View>

                <Text style={styles.llmExplanation}>{llmLabel.explanation}</Text>

                {llmLabel.top_tokens?.length > 0 && (
                  <View style={styles.tokenChips}>
                    {llmLabel.top_tokens.slice(0, 10).map((tok, i) => (
                      <View key={i} style={styles.chip}>
                        <Text style={styles.chipText}>{tok}</Text>
                      </View>
                    ))}
                  </View>
                )}

                {analysisMeta && (
                  <View style={styles.metaBox}>
                    <Text style={styles.metaText}>Sentences scanned: {analysisMeta.sentenceLimit}</Text>
                    <Text style={styles.metaText}>Min activation: {analysisMeta.minAct}</Text>
                    <Text style={styles.metaText}>Matches returned: {analysisMeta.totalMatches}</Text>
                    <Text style={styles.metaText}>Sentences used for LLM: {analysisMeta.corpusSentencesUsed}</Text>
                    <Text style={styles.metaText}>LLM example limit: {analysisMeta.llmExampleLimit}</Text>
                    <Text style={styles.metaText}>Activation cache: {analysisMeta.activationCacheHit ? 'hit' : 'miss'}</Text>
                    <Text style={styles.metaText}>Label cache: {analysisMeta.labelCacheHit ? 'hit' : 'miss'}</Text>
                  </View>
                )}
              </View>
            )}
          </View>

          {activationPayload?.matches?.length > 0 && (
            <View style={styles.activationsListWrap}>
              <Text style={styles.subheading}>Top Activation Matches</Text>
              {activationPayload.matches.slice(0, 25).map((match, idx) => (
                <View key={`match-${idx}`} style={styles.matchCard}>
                  <View style={styles.matchMetaRow}>
                    <Text style={styles.matchMetaText}>
                      Sentence #{match.sentence_index} | Token #{match.token_index}
                    </Text>
                    <Text style={styles.matchActivation}>{Number(match.activation || 0).toFixed(4)}</Text>
                  </View>

                  <Text style={styles.matchContext}>
                    {match.left_context}
                    <Text style={styles.matchToken}>{match.token}</Text>
                    {match.right_context}
                  </Text>

                  <Text style={styles.matchSentence}>{match.sentence}</Text>
                </View>
              ))}
            </View>
          )}
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  topBar: {
    padding: 10,
    flexDirection: 'row',
    justifyContent: 'flex-start',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
    backgroundColor: '#f8f9fa',
  },
  doneButton: {
    padding: 8,
    backgroundColor: '#e2e8f0',
    borderRadius: 6,
  },
  doneButtonText: {
    fontWeight: '600',
    color: '#475569',
  },
  contentScroll: {
    flex: 1,
  },
  mainContent: {
    padding: 20,
    maxWidth: 1200,
    alignSelf: 'center',
    width: '100%',
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
    flexWrap: 'wrap',
    gap: 10,
  },
  pageTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#334155',
    flex: 1,
    minWidth: 300,
  },
  modelCard: {
    backgroundColor: '#eff6ff',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#bfdbfe',
    alignItems: 'flex-end',
  },
  modelInfo: { fontSize: 10, color: '#64748b' },
  modelDetail: { fontSize: 10, color: '#64748b' },
  modelIndex: { fontSize: 12, fontWeight: 'bold', color: '#334155' },

  analysisPanel: {
    backgroundColor: '#f8fafc',
    borderWidth: 1,
    borderColor: '#e2e8f0',
    borderRadius: 12,
    padding: 14,
    marginBottom: 16,
    gap: 10,
  },
  panelTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#0f172a',
  },
  panelHint: {
    color: '#475569',
    fontSize: 13,
    lineHeight: 20,
  },
  inputRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  inputField: {
    flex: 1,
    minWidth: 180,
  },
  inputLabel: {
    fontSize: 12,
    color: '#334155',
    marginBottom: 6,
    fontWeight: '600',
  },
  numericInput: {
    borderWidth: 1,
    borderColor: '#cbd5e1',
    borderRadius: 8,
    backgroundColor: '#fff',
    paddingVertical: 8,
    paddingHorizontal: 10,
    fontSize: 14,
    color: '#0f172a',
    outlineStyle: 'none',
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  loadingText: {
    color: '#334155',
    fontSize: 13,
  },

  llmSection: {
    marginBottom: 20,
    gap: 10,
  },
  llmButton: {
    backgroundColor: '#2563eb',
    alignSelf: 'flex-start',
  },
  llmError: {
    color: '#dc2626',
    fontSize: 13,
  },
  llmResult: {
    backgroundColor: '#eff6ff',
    borderRadius: 8,
    padding: 14,
    borderWidth: 1,
    borderColor: '#bfdbfe',
    gap: 6,
  },
  resultTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1e3a8a',
    marginBottom: 4,
  },
  llmLabelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    flexWrap: 'wrap',
  },
  llmLabelText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1e3a8a',
    flex: 1,
  },
  confidenceBadge: {
    borderRadius: 20,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  confHigh: { backgroundColor: '#bbf7d0' },
  confMed: { backgroundColor: '#fef08a' },
  confLow: { backgroundColor: '#fecaca' },
  confidenceText: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#334155',
  },
  llmExplanation: {
    fontSize: 13,
    color: '#475569',
    lineHeight: 20,
  },
  tokenChips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginTop: 4,
  },
  chip: {
    backgroundColor: '#dbeafe',
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderWidth: 1,
    borderColor: '#93c5fd',
  },
  chipText: {
    fontSize: 11,
    color: '#1e3a8a',
    fontWeight: '600',
  },
  metaBox: {
    marginTop: 8,
    backgroundColor: '#f8fafc',
    borderWidth: 1,
    borderColor: '#e2e8f0',
    borderRadius: 8,
    padding: 10,
    gap: 2,
  },
  metaText: {
    fontSize: 12,
    color: '#334155',
  },

  actionButton: {
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 8,
  },
  actionButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 13,
  },

  activationsListWrap: {
    marginBottom: 20,
  },
  subheading: {
    fontSize: 16,
    fontWeight: '700',
    color: '#0f172a',
    marginBottom: 10,
  },
  matchCard: {
    backgroundColor: '#f8fafc',
    borderRadius: 10,
    padding: 12,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  matchMetaRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  matchMetaText: {
    fontSize: 12,
    color: '#334155',
    fontWeight: '600',
  },
  matchActivation: {
    fontSize: 12,
    color: '#0f766e',
    fontWeight: '700',
  },
  matchContext: {
    flex: 1,
    fontSize: 14,
    color: '#1f2937',
    lineHeight: 22,
    marginBottom: 6,
  },
  matchToken: {
    backgroundColor: '#d1fae5',
    color: '#065f46',
    fontWeight: '700',
  },
  matchSentence: {
    fontSize: 12,
    lineHeight: 18,
    color: '#64748b',
  },
});
