import React, { useMemo, useState } from 'react';
import {
  ActivityIndicator, ScrollView, StyleSheet, Text,
  TextInput, TouchableOpacity, View, Platform,
} from 'react-native';

const API_BASE = 'http://localhost:8000';
const PEOPLE_SPEECH_DATASET = 'MLCommons/peoples_speech';
const PEOPLE_SPEECH_CONFIG = 'validation';
const PEOPLE_SPEECH_SPLIT = 'validation';

function sanitizeDecimalInput(value) {
  const cleaned = (value || '').replace(/[^0-9.]/g, '');
  const firstDotIndex = cleaned.indexOf('.');
  if (firstDotIndex === -1) return cleaned;
  return cleaned.slice(0, firstDotIndex + 1) + cleaned.slice(firstDotIndex + 1).replace(/\./g, '');
}

// ─── CYBERPUNK COLOR TOKENS ──────────────────────────────────────────────────
const DARK_THEME = {
  bg:          '#06090f',
  bgPanel:     '#0d1117',
  bgCard:      '#111827',
  bgCardHover: '#1a2435',
  border:      '#1e2d3d',
  cyan:        '#00ffcc',
  pink:        '#ff00cc',
  yellow:      '#ffcc00',
  blue:        '#00aaff',
  textPrimary: '#e0f0ff',
  textSecond:  '#9ab3cc',
  textMuted:   '#7e95ad',
  error:       '#ff3355',
  errorBg:     '#1a0010',
  confHigh:    '#00cc66',
  confMed:     '#ffcc00',
  confLow:     '#ff3355',
};

const LIGHT_THEME = {
  bg:          '#f1f1f1',
  bgPanel:     '#ffffff',
  bgCard:      '#fbfbfb',
  bgCardHover: '#f2f2f2',
  border:      '#d8d8d8',
  cyan:        '#d89b54',
  pink:        '#5f79c9',
  yellow:      '#d89b54',
  blue:        '#4a67bf',
  textPrimary: '#101317',
  textSecond:  '#1c222a',
  textMuted:   '#383f49',
  error:       '#c54141',
  errorBg:     '#fff1f1',
  confHigh:    '#1f7a3a',
  confMed:     '#9a6a17',
  confLow:     '#b73c3c',
};

const mono = Platform.OS === 'web' ? '"Courier New", monospace' : 'monospace';
const serif = Platform.OS === 'web' ? 'Georgia, serif' : undefined;

export default function FeatureDetails({ feature, onClose, modelId, themeMode = 'dark' }) {
  const theme = themeMode === 'light' ? LIGHT_THEME : DARK_THEME;
  const styles = useMemo(() => createStyles(theme), [themeMode]);
  const [maxSentences, setMaxSentences] = useState('120');
  const [activatingExamples, setActivatingExamples] = useState('25');
  const [minActivation, setMinActivation] = useState('0.1');
  const [pageSize] = useState(25);
  const [matchesPage, setMatchesPage] = useState(1);
  const [lastActivationQuery, setLastActivationQuery] = useState(null);

  const [llmLabel, setLlmLabel] = useState(null);
  const [llmLoading, setLlmLoading] = useState(false);
  const [pageLoading, setPageLoading] = useState(false);
  const [llmError, setLlmError] = useState(null);

  const [activationPayload, setActivationPayload] = useState(null);
  const [analysisMeta, setAnalysisMeta] = useState(null);

  const promptAlignedRows = (llmLabel?.llm_prompt_examples || []).map((ex, idx) => ({
    id: `prompt-${idx}`,
    context: ex.context,
    activation: ex.activation,
  }));
  const usePromptAlignedRows = llmLabel?.llm_prompt_examples?.length > 0;

  const loadActivationPage = async (query, pageNo) => {
    setPageLoading(true);
    try {
      const res = await fetch(`${API_BASE}/feature-activations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...query, page: pageNo, page_size: pageSize }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setActivationPayload(data);
      setMatchesPage(data.page || pageNo);
      return data;
    } finally {
      setPageLoading(false);
    }
  };

  const fetchLlmLabel = async () => {
    if (!feature?.id || !modelId) return;

    const sentenceLimit = maxSentences.trim() === '' ? null : Math.max(1, Number(maxSentences) || 0);
    const activatingExamplesLimit = activatingExamples.trim() === '' ? null : Math.max(1, Number(activatingExamples) || 0);
    const effectiveSentenceLimit = sentenceLimit ?? 200;
    const minAct = Math.max(0, Number(minActivation) || 0);

    setLlmLoading(true);
    setLlmLabel(null);
    setLlmError(null);
    setActivationPayload(null);
    setAnalysisMeta(null);

    try {
      const hasExplicitSentenceLimit = sentenceLimit !== null;
      const activationQuery = {
        model_id: modelId,
        feature_id: feature.id,
        dataset_name: PEOPLE_SPEECH_DATASET,
        dataset_config: PEOPLE_SPEECH_CONFIG,
        split: PEOPLE_SPEECH_SPLIT,
        max_sentences: hasExplicitSentenceLimit ? effectiveSentenceLimit : null,
        target_activating_examples: !hasExplicitSentenceLimit ? activatingExamplesLimit : null,
        min_activation: minAct,
        page: 1,
        page_size: pageSize,
      };

      let activationData = await loadActivationPage(activationQuery, 1);
      setLastActivationQuery(activationQuery);

      let uniqueSentences = activationData.activating_sentences || [];
      if (uniqueSentences.length === 0) throw new Error('No activating contexts found with the selected thresholds.');

      const llmContextCap = activatingExamplesLimit || 25;
      const corpusForLlm = uniqueSentences.slice(0, llmContextCap);
      setActivationPayload(activationData);

      const labelRes = await fetch(`${API_BASE}/label-feature`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelId,
          feature_idx: feature.id,
          analyzer: 'sae',
          corpus_texts: corpusForLlm,
          labeling_config: {
            top_k: llmContextCap,
            skip_first_token: true,
            min_activation: minAct,
            num_sentences: hasExplicitSentenceLimit ? effectiveSentenceLimit : effectiveSentenceLimit,
          },
        }),
      });

      if (!labelRes.ok) {
        const err = await labelRes.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${labelRes.status}`);
      }

      const labelData = await labelRes.json();
      setLlmLabel(labelData);
      setAnalysisMeta({
        requestId: labelData.request_id || null,
        llmActivationMode: labelData.llm_activation_mode || 'standardize',
        sentenceLimit: hasExplicitSentenceLimit ? effectiveSentenceLimit : null,
        scannedSentences: activationData.scanned_sentences,
        activatingExamplesLimit: activatingExamplesLimit,
        minAct,
        corpusSentencesUsed: corpusForLlm.length,
        totalMatches: activationData.total_matches || 0,
        totalPages: activationData.total_pages || 1,
        dataset: `${PEOPLE_SPEECH_DATASET} [${PEOPLE_SPEECH_SPLIT}]`,
      });
    } catch (e) {
      setLlmError(e.message);
    } finally {
      setLlmLoading(false);
    }
  };

  const confidenceColor = (conf) => {
    if (conf === 'high') return theme.confHigh;
    if (conf === 'medium') return theme.confMed;
    return theme.confLow;
  };

  return (
    <View style={styles.container}>
      {/* Top bar */}
      <View style={styles.topBar}>
        <TouchableOpacity style={styles.backButton} onPress={onClose}>
          <Text style={styles.backButtonText}>← BACK</Text>
        </TouchableOpacity>
        <View style={styles.topBarCenter}>
          <Text style={styles.topBarLabel}>FEATURE ANALYSIS</Text>
          <Text style={styles.topBarId}>#{feature?.id}</Text>
        </View>
        <View style={styles.topBarRight}>
          <Text style={styles.topBarModel}>GPT-2 SAE</Text>
        </View>
      </View>

      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        <View style={styles.content}>

          {/* Feature header */}
          <View style={styles.featureHeader}>
            <View style={styles.featureIdBlock}>
              <Text style={styles.featureIdLabel}>// FEATURE ID</Text>
              <Text style={styles.featureIdNum}>#{feature?.id ?? '—'}</Text>
            </View>
            <View style={styles.featureDescBlock}>
              <Text style={styles.featureDescLabel}>DESCRIPTION</Text>
              <Text style={styles.featureDescText}>{feature?.description || 'No description available'}</Text>
            </View>
          </View>

          {/* Analysis config */}
          <View style={styles.configPanel}>
            <View style={styles.sectionLabelRow}>
              <View style={styles.sectionDot} />
              <Text style={styles.sectionLabel}>ANALYSIS PARAMETERS</Text>
            </View>
            <Text style={styles.configHint}>
              Dataset: MLCommons/peoples_speech · Scans sentences to find activating contexts, then sends them to LLM for labeling.
            </Text>

            <View style={styles.inputGrid}>
              {[
                { label: 'SENTENCES TO SCAN', value: maxSentences, onChange: (t) => setMaxSentences(t.replace(/[^0-9]/g, '')), placeholder: '120' },
                { label: 'EXAMPLES FOR LLM', value: activatingExamples, onChange: (t) => setActivatingExamples(t.replace(/[^0-9]/g, '')), placeholder: '25' },
                { label: 'MIN ACTIVATION', value: minActivation, onChange: (t) => setMinActivation(sanitizeDecimalInput(t)), placeholder: '0.1' },
              ].map(field => (
                <View key={field.label} style={styles.inputField}>
                  <Text style={styles.inputLabel}>{field.label}</Text>
                  <TextInput
                    style={styles.numericInput}
                    value={field.value}
                    onChangeText={field.onChange}
                    keyboardType="numeric"
                    placeholder={field.placeholder}
                    placeholderTextColor={theme.textMuted}
                  />
                </View>
              ))}
              <View style={styles.inputField}>
                <Text style={styles.inputLabel}>PAGE SIZE</Text>
                <View style={styles.pageInfoBox}>
                  <Text style={styles.pageInfoText}>{pageSize} per page</Text>
                </View>
              </View>
            </View>

            <TouchableOpacity
              style={[styles.runButton, llmLoading && styles.runButtonDisabled]}
              onPress={fetchLlmLabel}
              disabled={llmLoading}
            >
              {llmLoading
                ? <><ActivityIndicator color="#000" size="small" /><Text style={styles.runButtonText}> ANALYZING...</Text></>
                : <Text style={styles.runButtonText}>▶ ANALYZE FEATURE #{feature?.id}</Text>
              }
            </TouchableOpacity>

            {(llmLoading || pageLoading) && (
              <View style={styles.loadingRow}>
                <View style={styles.loadingDot} />
                <Text style={styles.loadingText}>
                  {llmLoading
                    ? 'Collecting activation contexts and querying LLM...'
                    : 'Loading page...'}
                </Text>
              </View>
            )}
          </View>

          {/* Error */}
          {llmError && (
            <View style={styles.errorBox}>
              <Text style={styles.errorLabel}>⚠ ERROR</Text>
              <Text style={styles.errorText}>{llmError}</Text>
            </View>
          )}

          {/* LLM Result */}
          {llmLabel && (
            <View style={styles.resultPanel}>
              <View style={styles.sectionLabelRow}>
                <View style={[styles.sectionDot, { backgroundColor: theme.pink }]} />
                <Text style={[styles.sectionLabel, { color: theme.pink }]}>LLM LABEL RESULT</Text>
                <View style={[styles.confidenceBadge, { backgroundColor: confidenceColor(llmLabel.confidence) + '33', borderColor: confidenceColor(llmLabel.confidence) + '88' }]}>
                  <Text style={[styles.confidenceBadgeText, { color: confidenceColor(llmLabel.confidence) }]}>
                    {llmLabel.confidence?.toUpperCase() ?? '—'}
                  </Text>
                </View>
              </View>

              <Text style={styles.llmLabelText}>{llmLabel.label || '(no label returned)'}</Text>
              <Text style={styles.llmExplanation}>{llmLabel.explanation}</Text>

              {llmLabel.top_tokens?.length > 0 && (
                <View>
                  <Text style={styles.topTokensLabel}>TOP TOKENS</Text>
                  <View style={styles.tokenChipRow}>
                    {llmLabel.top_tokens.slice(0, 10).map((tok, i) => (
                      <View key={i} style={styles.tokenChip}>
                        <Text style={styles.tokenChipText}>{tok}</Text>
                      </View>
                    ))}
                  </View>
                </View>
              )}

              {analysisMeta && (
                <View style={styles.metaBox}>
                  {[
                    ['DATASET', analysisMeta.dataset],
                    ['REQUEST ID', analysisMeta.requestId || 'n/a'],
                    ['LLM MODE', analysisMeta.llmActivationMode],
                    ['SENTENCES REQUESTED', analysisMeta.sentenceLimit ?? 'auto'],
                    ['SENTENCES SCANNED', analysisMeta.scannedSentences],
                    ['EXAMPLES REQUESTED', analysisMeta.activatingExamplesLimit ?? 'all'],
                    ['MIN ACTIVATION', analysisMeta.minAct],
                    ['TOTAL MATCHES', analysisMeta.totalMatches],
                    ['TOTAL PAGES', analysisMeta.totalPages],
                    ['SENT TO LLM', analysisMeta.corpusSentencesUsed],
                  ].map(([key, val]) => (
                    <View key={key} style={styles.metaRow}>
                      <Text style={styles.metaKey}>{key}</Text>
                      <Text style={styles.metaVal}>{String(val)}</Text>
                    </View>
                  ))}
                </View>
              )}

              {llmLabel.llm_prompt_examples?.length > 0 && (
                <View style={styles.metaBox}>
                  <Text style={[styles.topTokensLabel, { marginBottom: 8 }]}>EXAMPLES SENT TO LLM</Text>
                  {llmLabel.llm_prompt_examples.slice(0, 12).map((ex, i) => (
                    <View key={i} style={styles.promptExampleRow}>
                      <Text style={styles.promptExampleNum}>{i + 1}.</Text>
                      <Text style={styles.promptExampleAct}>[{Number(ex.activation || 0).toFixed(3)}]</Text>
                      <Text style={styles.promptExampleContext}>{ex.context}</Text>
                    </View>
                  ))}
                </View>
              )}
            </View>
          )}

          {/* Match list */}
          {(usePromptAlignedRows || activationPayload?.matches?.length > 0) && (
            <View style={styles.matchesPanel}>
              <View style={styles.sectionLabelRow}>
                <View style={[styles.sectionDot, { backgroundColor: theme.yellow }]} />
                <Text style={[styles.sectionLabel, { color: theme.yellow }]}>
                  {usePromptAlignedRows ? 'TOP ACTIVATION MATCHES (PROMPT-ALIGNED)' : 'TOP ACTIVATION MATCHES'}
                </Text>
              </View>

              {usePromptAlignedRows
                ? promptAlignedRows.map((row, idx) => (
                    <View key={row.id} style={styles.matchCard}>
                      <View style={styles.matchCardHeader}>
                        <Text style={styles.matchCardMeta}>EXAMPLE #{idx + 1}</Text>
                        <View style={styles.activationBadge}>
                          <Text style={styles.activationBadgeText}>{Number(row.activation || 0).toFixed(4)}</Text>
                        </View>
                      </View>
                      <Text style={styles.matchContext}>{row.context}</Text>
                    </View>
                  ))
                : activationPayload.matches.map((match, idx) => (
                    <View key={idx} style={styles.matchCard}>
                      <View style={styles.matchCardHeader}>
                        <Text style={styles.matchCardMeta}>
                          SEN #{match.sentence_index} · TOK #{match.token_index}
                        </Text>
                        <View style={styles.activationBadge}>
                          <Text style={styles.activationBadgeText}>{Number(match.activation || 0).toFixed(4)}</Text>
                        </View>
                      </View>
                      <Text style={styles.matchContext}>
                        {match.left_context}
                        <Text style={styles.matchToken}>{match.token}</Text>
                        {match.right_context}
                      </Text>
                      <Text style={styles.matchSentence}>{match.sentence}</Text>
                    </View>
                  ))
              }
            </View>
          )}
        </View>
      </ScrollView>

      {/* Pagination bar */}
      {activationPayload?.matches?.length > 0 && !usePromptAlignedRows && (
        <View style={styles.paginationBar}>
          <TouchableOpacity
            style={[styles.pageButton, (!activationPayload?.has_prev_page || llmLoading || pageLoading) && styles.pageButtonDisabled]}
            disabled={!activationPayload?.has_prev_page || llmLoading || pageLoading}
            onPress={async () => {
              if (!lastActivationQuery || !activationPayload?.has_prev_page) return;
              try { await loadActivationPage(lastActivationQuery, Math.max(1, matchesPage - 1)); }
              catch (e) { setLlmError(e.message); }
            }}
          >
            <Text style={styles.pageButtonText}>← PREV</Text>
          </TouchableOpacity>

          <View style={styles.pageIndicator}>
            <Text style={styles.pageIndicatorText}>
              PAGE {activationPayload.page} / {activationPayload.total_pages}
            </Text>
          </View>

          <TouchableOpacity
            style={[styles.pageButton, (!activationPayload?.has_next_page || llmLoading || pageLoading) && styles.pageButtonDisabled]}
            disabled={!activationPayload?.has_next_page || llmLoading || pageLoading}
            onPress={async () => {
              if (!lastActivationQuery || !activationPayload?.has_next_page) return;
              try { await loadActivationPage(lastActivationQuery, matchesPage + 1); }
              catch (e) { setLlmError(e.message); }
            }}
          >
            <Text style={styles.pageButtonText}>NEXT →</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const createStyles = (C) => StyleSheet.create({
  container: { flex: 1, backgroundColor: C.bg },
  scrollView: { flex: 1 },

  // Top bar
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 12,
    backgroundColor: C.bgPanel,
    borderBottomWidth: 1,
    borderBottomColor: C.cyan,
  },
  backButton: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderColor: C.cyan + '66',
    borderRadius: 2,
  },
  backButtonText: {
    fontFamily: mono,
    fontSize: 10,
    fontWeight: '700',
    color: C.cyan,
    letterSpacing: 2,
  },
  topBarCenter: { alignItems: 'center' },
  topBarLabel: {
    fontFamily: mono,
    fontSize: 9,
    color: C.textMuted,
    letterSpacing: 3,
    marginBottom: 2,
  },
  topBarId: {
    fontFamily: mono,
    fontSize: 18,
    fontWeight: '900',
    color: C.cyan,
    letterSpacing: 2,
  },
  topBarRight: { alignItems: 'flex-end' },
  topBarModel: {
    fontFamily: mono,
    fontSize: 9,
    color: C.textSecond,
    letterSpacing: 2,
  },

  content: {
    padding: 20,
    paddingBottom: 80,
    maxWidth: 1000,
    alignSelf: 'center',
    width: '100%',
  },

  // Feature header
  featureHeader: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 20,
    flexWrap: 'wrap',
  },
  featureIdBlock: {
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.cyan + '55',
    borderRadius: 4,
    padding: 16,
    alignItems: 'center',
    minWidth: 120,
  },
  featureIdLabel: {
    fontFamily: mono,
    fontSize: 9,
    color: C.textMuted,
    letterSpacing: 2,
    marginBottom: 8,
  },
  featureIdNum: {
    fontFamily: mono,
    fontSize: 28,
    fontWeight: '900',
    color: C.cyan,
  },
  featureDescBlock: {
    flex: 1,
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.border,
    borderLeftWidth: 3,
    borderLeftColor: C.cyan,
    borderRadius: 4,
    padding: 16,
    minWidth: 200,
  },
  featureDescLabel: {
    fontFamily: mono,
    fontSize: 9,
    color: C.textMuted,
    letterSpacing: 2,
    marginBottom: 8,
  },
  featureDescText: {
    fontSize: 15,
    color: C.textPrimary,
    lineHeight: 24,
    fontFamily: serif,
  },

  // Config panel
  configPanel: {
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 4,
    padding: 16,
    marginBottom: 16,
  },
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
    fontFamily: mono,
    fontSize: 10,
    fontWeight: '700',
    color: C.cyan,
    letterSpacing: 3,
  },
  configHint: {
    fontFamily: mono,
    fontSize: 11,
    color: C.textSecond,
    lineHeight: 18,
    marginBottom: 14,
  },
  inputGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 14,
  },
  inputField: { flex: 1, minWidth: 150 },
  inputLabel: {
    fontFamily: mono,
    fontSize: 9,
    color: C.textSecond,
    letterSpacing: 2,
    marginBottom: 6,
  },
  numericInput: {
    backgroundColor: C.bgPanel,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 2,
    paddingVertical: 8,
    paddingHorizontal: 10,
    fontSize: 14,
    color: C.cyan,
    fontFamily: mono,
    outlineStyle: 'none',
  },
  pageInfoBox: {
    backgroundColor: C.bgPanel,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 2,
    paddingVertical: 8,
    paddingHorizontal: 10,
  },
  pageInfoText: {
    fontFamily: mono,
    fontSize: 12,
    color: C.textMuted,
  },

  runButton: {
    backgroundColor: C.cyan,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 11,
    paddingHorizontal: 20,
    borderRadius: 2,
    alignSelf: 'flex-start',
    gap: 8,
  },
  runButtonDisabled: { opacity: 0.5 },
  runButtonText: {
    fontFamily: mono,
    fontWeight: '900',
    fontSize: 12,
    color: '#000',
    letterSpacing: 2,
  },

  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginTop: 12,
  },
  loadingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: C.cyan,
    opacity: 0.7,
  },
  loadingText: {
    fontFamily: mono,
    fontSize: 11,
    color: C.textSecond,
  },

  // Error
  errorBox: {
    backgroundColor: C.errorBg,
    borderWidth: 1,
    borderColor: C.error + '66',
    borderRadius: 4,
    padding: 12,
    marginBottom: 16,
  },
  errorLabel: {
    fontFamily: mono,
    fontSize: 9,
    fontWeight: '700',
    color: C.error,
    letterSpacing: 2,
    marginBottom: 4,
  },
  errorText: {
    fontFamily: mono,
    fontSize: 12,
    color: C.error + 'cc',
    lineHeight: 18,
  },

  // LLM result panel
  resultPanel: {
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.pink + '44',
    borderLeftWidth: 3,
    borderLeftColor: C.pink,
    borderRadius: 4,
    padding: 16,
    marginBottom: 16,
    gap: 12,
  },
  confidenceBadge: {
    marginLeft: 'auto',
    borderWidth: 1,
    borderRadius: 2,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  confidenceBadgeText: {
    fontFamily: mono,
    fontSize: 9,
    fontWeight: '700',
    letterSpacing: 2,
  },
  llmLabelText: {
    fontSize: 18,
    fontWeight: '700',
    color: C.textPrimary,
    lineHeight: 26,
    fontFamily: serif,
  },
  llmExplanation: {
    fontSize: 13,
    color: C.textSecond,
    lineHeight: 20,
    fontFamily: serif,
  },
  topTokensLabel: {
    fontFamily: mono,
    fontSize: 9,
    color: C.textMuted,
    letterSpacing: 2,
    marginBottom: 6,
  },
  tokenChipRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  tokenChip: {
    backgroundColor: C.blue + '22',
    borderWidth: 1,
    borderColor: C.blue + '55',
    borderRadius: 2,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  tokenChipText: {
    fontFamily: mono,
    fontSize: 11,
    fontWeight: '600',
    color: C.blue,
  },

  // Meta box
  metaBox: {
    backgroundColor: C.bgPanel,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 4,
    padding: 12,
    gap: 4,
  },
  metaRow: {
    flexDirection: 'row',
    gap: 10,
    alignItems: 'flex-start',
  },
  metaKey: {
    fontFamily: mono,
    fontSize: 9,
    color: C.textMuted,
    letterSpacing: 1,
    minWidth: 160,
  },
  metaVal: {
    fontFamily: mono,
    fontSize: 11,
    color: C.textSecond,
    flex: 1,
  },

  // Prompt examples
  promptExampleRow: {
    flexDirection: 'row',
    gap: 8,
    paddingVertical: 4,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
  },
  promptExampleNum: {
    fontFamily: mono,
    fontSize: 10,
    color: C.textMuted,
    minWidth: 18,
  },
  promptExampleAct: {
    fontFamily: mono,
    fontSize: 10,
    color: C.yellow,
    minWidth: 52,
  },
  promptExampleContext: {
    flex: 1,
    fontSize: 12,
    color: C.textSecond,
    lineHeight: 18,
  },

  // Match cards
  matchesPanel: {
    gap: 8,
    marginBottom: 20,
  },
  matchCard: {
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 4,
    padding: 12,
  },
  matchCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  matchCardMeta: {
    fontFamily: mono,
    fontSize: 9,
    color: C.textMuted,
    letterSpacing: 1,
  },
  activationBadge: {
    backgroundColor: C.yellow + '22',
    borderWidth: 1,
    borderColor: C.yellow + '55',
    borderRadius: 2,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  activationBadgeText: {
    fontFamily: mono,
    fontSize: 10,
    fontWeight: '700',
    color: C.yellow,
  },
  matchContext: {
    fontSize: 14,
    color: C.textSecond,
    lineHeight: 22,
    marginBottom: 6,
  },
  matchToken: {
    backgroundColor: C.cyan + '33',
    color: C.cyan,
    fontWeight: '700',
  },
  matchSentence: {
    fontFamily: mono,
    fontSize: 11,
    color: C.textMuted,
    lineHeight: 18,
  },

  // Pagination
  paginationBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 12,
    backgroundColor: C.bgPanel,
    borderTopWidth: 1,
    borderTopColor: C.border,
  },
  pageButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: C.cyan + '66',
    borderRadius: 2,
    backgroundColor: C.cyan + '11',
  },
  pageButtonDisabled: { opacity: 0.3 },
  pageButtonText: {
    fontFamily: mono,
    fontSize: 10,
    fontWeight: '700',
    color: C.cyan,
    letterSpacing: 2,
  },
  pageIndicator: {
    backgroundColor: C.bgCard,
    borderWidth: 1,
    borderColor: C.border,
    borderRadius: 2,
    paddingHorizontal: 14,
    paddingVertical: 6,
  },
  pageIndicatorText: {
    fontFamily: mono,
    fontSize: 10,
    color: C.textSecond,
    letterSpacing: 2,
  },
});