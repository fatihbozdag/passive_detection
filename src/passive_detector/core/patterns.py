#!/usr/bin/env python3
"""
Pattern definitions for passive voice detection.

This module contains regex patterns and word lists used to identify and filter passive voice
constructions in text. These patterns are used by the passive detector to accurately
identify passive voice while avoiding common false positives.
"""

# Basic passive voice patterns
PASSIVE_PATTERNS = {
    'basic': r'\b(am|is|are|was|were|be|been|being)\s+(\w+ed|\w+en|\w+t)\b',
    'perfect': r'\b(have|has|had)\s+been\s+(\w+ed|\w+en|\w+t)\b',
    'modal': r'\b(will|would|shall|should|may|might|must|can|could)\s+be\s+(\w+ed|\w+en|\w+t)\b',
    'get': r'\b(get|gets|got|gotten)\s+(\w+ed|\w+en|\w+t)\b',
}

# Common passive expressions with high confidence
COMMON_PASSIVE_PATTERNS = [
    # Election/selection patterns - Very reliable passive indicators
    r'\b(was|were|is|are|be|been)\s+(elected|chosen|selected|appointed|nominated|voted)\b',
    # Event patterns - Very reliable passive indicators
    r'\b(was|were|is|are|be|been)\s+(held|scheduled|organized|canceled|cancelled|postponed)\b',
    # Delivery patterns - Very reliable passive indicators
    r'\b(was|were|is|are|be|been)\s+(delivered|shipped|sent|given|handed|distributed)\b',
    # Creation patterns - Very reliable passive indicators
    r'\b(was|were|is|are|be|been)\s+(made|created|built|constructed|developed|produced)\b',
    # Perception patterns - Very reliable passive indicators
    r'\b(was|were|is|are|be|been)\s+(seen|heard|watched|viewed|observed|noticed)\b',
    # Communication patterns - Very reliable passive indicators
    r'\b(was|were|is|are|be|been)\s+(told|asked|informed|notified|advised|instructed)\b',
    # Modification patterns - Very reliable passive indicators
    r'\b(was|were|is|are|be|been)\s+(changed|modified|altered|adjusted|adapted|revised)\b',
    # Play/Performance patterns - Common in sports/entertainment
    r'\b(was|were|is|are|be|been)\s+(played|performed|conducted|executed|acted)\b',
    # Halting/Stopping patterns - Common in event/activity descriptions
    r'\b(was|were|is|are|be|been)\s+(halted|stopped|paused|suspended|interrupted)\b',
    # Discussion/Analysis patterns - Common in academic/meeting contexts
    r'\b(was|were|is|are|be|been)\s+(discussed|analyzed|examined|reviewed|considered)\b',
    # Legal/formal patterns - Common in legal/formal descriptions
    r'\b(was|were|is|are|be|been)\s+(convicted|sentenced|fined|charged|indicted|acquitted)\b',
]

# Specific patterns for commonly missed passives
SPECIAL_PASSIVE_PATTERNS = {
    'election': r'\b(was|were|is|are|been)\s+(elected|chosen|selected|appointed|voted)\b',
    'event': r'\b(was|were|is|are|been)\s+(held|organized|cancelled|canceled|postponed|scheduled)\b',
    'by_agent': r'\b(was|were|is|are|been)\s+(\w+ed|\w+en|\w+t)\s+by\b',
    'negated': r"\b(isn't|aren't|wasn't|weren't|isn|aren|wasn|weren)\s+(\w+ed|\w+en|\w+t)\b",
    'left': r'\b(was|were|is|are|been)\s+left\s+(\w+ed|\w+en|\w+t)\b',
    'simple': r'\b(politics|sports|football|basketball|baseball|soccer|games|science|research|discoveries|studies)\s+(are|is|were|was)\s+(\w+ed|\w+en|\w+t)\b',
}

# Fixed expressions that often indicate passive voice but may not follow standard patterns
PASSIVE_EXPRESSIONS = [
    'supposed to',
    'meant to',
    'required to',
    'expected to',
    'asked to',
    'told to',
    'made to',
    'forced to',
    'allowed to',
]

# Combinations that look like passive voice but should not be counted as such
NON_PASSIVE_COMBOS = [
    'is able', 'was able', 'are able', 'were able',
    'is about', 'was about', 'are about', 'were about',
    'is available', 'was available', 'are available', 'were available',
    'is possible', 'was possible', 'are possible', 'were possible',
    'is responsible', 'was responsible', 'are responsible', 'were responsible',
    # Add more common non-passive combinations
    'is important', 'was important', 'are important', 'were important',
    'is difficult', 'was difficult', 'are difficult', 'were difficult',
    'is easy', 'was easy', 'are easy', 'were easy',
    'is great', 'was great', 'are great', 'were great',
    'is good', 'was good', 'are good', 'were good',
    'is hot', 'was hot', 'are hot', 'were hot',
    'is cold', 'was cold', 'are cold', 'were cold',
    'is filled', 'was filled', 'are filled', 'were filled',
    'is based', 'was based', 'are based', 'were based',
    'is involved', 'was involved', 'are involved', 'were involved',
    'is located', 'was located', 'are located', 'were located',
    'is situated', 'was situated', 'are situated', 'were situated',
    'is part', 'was part', 'are part', 'were part',
    'is best', 'was best', 'are best', 'were best',
    'is favorite', 'was favorite', 'are favorite', 'were favorite',
    'is better', 'was better', 'are better', 'were better',
    'is worse', 'was worse', 'are worse', 'were worse',
    'is needed', 'was needed', 'are needed', 'were needed',
    'is required', 'was required', 'are required', 'were required',
]

# Common false positive patterns in passive detection
FALSE_POSITIVE_PATTERNS = [
    # Emotional/mental states
    r'\b(am|is|are)\s+(stunned|surprised|shocked)\b',
    # Adjectival states
    r'\b(is|are|was|were)\s+(heated|excited|easy|fun|paramount|full|ready|stressful)\b',
    # Filled as adjective
    r'\b(is|are)\s+filled\b',
    # Active voice constructions
    r'\bis\s+having\b',
    # Can be + non-passive states
    r'\bcan\s+be\s+(stressful|difficult|easy|hard|challenging|exciting|fun)\b',
    # Going to constructions
    r'\bis\s+going\s+to\b',
    # Copular with non-passive states
    r'\b(is|are|was|were)\s+(just|still|also|now|here|there|about|almost|nearly|already|not|never)\b',
    # Subject + copula patterns
    r'\b(politics|science|culture|business|sports)\s+(is|are|was|were)\s+\w+\b',
    # Idiomatic expressions
    r'\b(is|are|was|were)\s+\w+\s+(topic|issue|career|subject|convenience|exercise|benefit|advantage)\b',
]

# Personal mental state verbs that often appear in passive-like constructions
PERSONAL_MENTAL_STATES = [
    'amazed', 'amused', 'annoyed', 'astonished', 'concerned',
    'confused', 'disappointed', 'discouraged', 'disgusted',
    'excited', 'exhausted', 'frightened', 'interested',
    'pleased', 'satisfied', 'shocked', 'surprised', 'tired',
    'worried'
]

# Adjectival participles that should not be counted as passive
ADJECTIVAL_PARTICIPLES = [
    'advanced', 'animated', 'authorized', 'automated',
    'balanced', 'calculated', 'calibrated', 'certified',
    'civilized', 'collected', 'committed', 'complicated',
    'concentrated', 'controlled', 'coordinated',
    'dedicated', 'designed', 'detailed', 'determined',
    'documented', 'educated', 'elevated', 'enhanced',
    'established', 'experienced', 'focused', 'illustrated',
    'integrated', 'interested', 'limited', 'located', 
    'marked', 'motivated', 'organized', 'oriented',
    'prepared', 'qualified', 'related', 'reserved',
    'sophisticated', 'structured', 'trained', 'trusted',
    # Adding more common adjectival participles to reduce false positives
    'filled', 'based', 'involved', 'learned', 'known',
    'specialized', 'chosen', 'inspired', 'driven', 'born',
    'proven', 'approved', 'selected', 'hidden', 
    'informed', 'accepted', 'united', 'valued',
    'excited', 'supposed', 'considered', 'understood', 'centered',
    'defined', 'entitled', 'inclined', 'aligned', 'recommended',
    'satisfied', 'respected', 'retired', 'scared', 'skilled',
    'talented', 'tempted', 'touched', 'unanswered', 'varied',
    'versed', 'welcomed', 'worried', 'written', 'unprepared',
    'unrelated', 'unused', 'unopened', 'unsolicited', 'untreated',
    'seen', 'heard', 'gone', 'done', 'said', 'spoken'
]

# Common verbs in passive constructions
COMMON_PASSIVE_VERBS = [
    'made', 'done', 'given', 'taken', 'seen', 'found', 'used',
    'held', 'known', 'shown', 'called', 'told', 'asked', 'left',
    'created', 'built', 'written', 'published', 'considered',
    'required', 'needed', 'expected', 'allowed', 'produced',
    'presented', 'performed', 'designed', 'developed', 'established',
    'implemented', 'organized', 'followed', 'supported', 'identified',
    'determined', 'affected', 'selected', 'measured', 'reported',
    'observed', 'evaluated', 'elected', 'processed', 'analyzed',
    'accepted', 'reviewed', 'recognized', 'provided', 'scheduled'
]

# Confidence levels for different pattern types
CONFIDENCE_SCORES = {
    'by_agent': 0.95,       # Passive with by-agent is highly confident
    'election_passive': 0.95,  # Election-related passives are very reliable
    'event_passive': 0.90,     # Event-related passives are reliable
    'high_confidence': 0.85,   # Custom high confidence patterns
    'dependency': 0.80,        # spaCy dependency parsed passives
    'perfect': 0.80,           # Perfect passives (have been + participle)
    'basic': 0.70,             # Basic passive form (is/was + participle)
    'expression': 0.75,        # Common passive expressions
    'modal': 0.65,             # Modal passives (can be seen, etc.)
    'get': 0.60,               # Get passives (got taken, etc.)
} 