import numpy as np
from typing import List, Dict, Tuple, Optional
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import Configuration


class QCSA:
    def __init__(self, min_samples: int = 5):
        self.min_samples = min_samples
        self.query_times_matrix: Optional[np.ndarray] = None
        self.query_cv: Optional[Dict[str, float]] = None
        self.csq_queries: List[str] = []
        self.ciq_queries: List[str] = []
        
    def extract_samples_from_history(self, 
                                     history: History,
                                     all_queries: List[str]) -> Dict[str, List[float]]:
        logger.info(f"QCSA: Extracting samples from History ({len(history)} observations)...")
        
        query_times: Dict[str, List[float]] = {q: [] for q in all_queries}
        
        for obs in history.observations:
            extra_info = getattr(obs, 'extra_info', {}) or {}
            qt_times = extra_info.get('qt_time', {})
            
            for query in all_queries:
                if query in qt_times:
                    time_val = qt_times[query]
                    if np.isfinite(time_val):
                        query_times[query].append(time_val)
                    else:
                        query_times[query].append(float('inf'))
                else:
                    query_times[query].append(float('inf'))
        
        valid_samples = sum(len([t for t in times if np.isfinite(t)]) for times in query_times.values())
        logger.info(f"QCSA: Extracted {len(history)} samples, {valid_samples} valid query-time pairs")
        
        return query_times
    
    def compute_cv(self, query_times: Dict[str, List[float]]) -> Dict[str, float]:
        query_cv = {}
        
        for query, times in query_times.items():
            valid_times = [t for t in times if np.isfinite(t)]
            
            if len(valid_times) < 2:
                query_cv[query] = 0.0
                logger.warning(f"QCSA: Query {query} has insufficient valid samples ({len(valid_times)})")
                continue
            
            mean_time = np.mean(valid_times)
            std_time = np.std(valid_times)
            
            if mean_time == 0:
                query_cv[query] = 0.0
            else:
                query_cv[query] = std_time / mean_time
        
        return query_cv
    
    def classify_queries(self, query_cv: Dict[str, float]) -> Tuple[List[str], List[str]]:
        if not query_cv:
            logger.warning("QCSA: No CV values to classify")
            return [], []
        
        cv_values = np.array(list(query_cv.values()))
        valid_cv = cv_values[np.isfinite(cv_values)]
        
        if len(valid_cv) == 0:
            logger.warning("QCSA: No valid CV values")
            return [], []
        
        max_cv = np.max(valid_cv)
        min_cv = np.min(valid_cv)
        width_cv = (max_cv - min_cv) / 3.0
        
        if width_cv == 0:
            logger.warning("QCSA: All CV values are the same, considering all queries as CSQ")
            return list(query_cv.keys()), []
        
        ciq_threshold = min_cv + width_cv
        
        csq_queries = []
        ciq_queries = []
        
        for query, cv in query_cv.items():
            if not np.isfinite(cv):
                ciq_queries.append(query)
            elif cv < ciq_threshold:
                ciq_queries.append(query)
            else:
                csq_queries.append(query)
        
        logger.info(f"QCSA: Classified {len(csq_queries)} CSQ and {len(ciq_queries)} CIQ queries")
        logger.info(f"QCSA: CV range: [{min_cv:.4f}, {max_cv:.4f}], CIQ threshold: {ciq_threshold:.4f}")
        
        return csq_queries, ciq_queries
    
    def analyze(self,
                history: History,
                all_queries: List[str]) -> Tuple[List[str], List[str]]:
        logger.info("QCSA: Starting query configuration sensitivity analysis...")
        
        if len(history) < self.min_samples:
            logger.warning(f"QCSA: Insufficient samples in History ({len(history)} < {self.min_samples}). "
                         f"Using all queries as CSQ.")
            self.query_cv = {}
            self.csq_queries = all_queries
            self.ciq_queries = []
            return all_queries, []
        
        query_times = self.extract_samples_from_history(history, all_queries)
        query_cv = self.compute_cv(query_times)
        self.query_cv = query_cv
        
        csq_queries, ciq_queries = self.classify_queries(query_cv)
        self.csq_queries = csq_queries
        self.ciq_queries = ciq_queries
        
        logger.info("QCSA: Analysis complete")
        logger.info(f"QCSA: CSQ queries ({len(csq_queries)}): {csq_queries[:10]}{'...' if len(csq_queries) > 10 else ''}")
        logger.info(f"QCSA: CIQ queries ({len(ciq_queries)}): {ciq_queries[:10]}{'...' if len(ciq_queries) > 10 else ''}")
        
        return csq_queries, ciq_queries
    
    def get_rqa_queries(self) -> List[str]:
        return self.csq_queries.copy()
    
    def get_analysis_info(self) -> Dict:
        return {
            'min_samples': self.min_samples,
            'query_cv': self.query_cv,
            'csq_queries': self.csq_queries,
            'ciq_queries': self.ciq_queries,
            'num_csq': len(self.csq_queries),
            'num_ciq': len(self.ciq_queries),
        }
