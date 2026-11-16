-- sql/example_queries.sql
-- Useful questions you can answer in MySQL Workbench once analyses are stored
-- (python main.py some_file.csv --target some_column --save).

USE datatrust;

-- 1. Latest score for every dataset (window function: newest run per dataset)
SELECT name, analyzed_at, trust_score, trust_grade, ml_score, ml_grade, total_issues
FROM (
    SELECT d.name, r.*,
           ROW_NUMBER() OVER (PARTITION BY r.dataset_id ORDER BY r.analyzed_at DESC, r.id DESC) AS rn
    FROM analysis_runs r
    JOIN datasets d ON d.id = r.dataset_id
) latest
WHERE rn = 1
ORDER BY trust_score;

-- 2. Score trend for one dataset, with the change since the previous run (LAG)
SELECT r.id AS run_id, r.analyzed_at, r.trust_score,
       r.trust_score - LAG(r.trust_score) OVER (ORDER BY r.analyzed_at, r.id) AS trust_change,
       r.file_sha256 <> LAG(r.file_sha256) OVER (ORDER BY r.analyzed_at, r.id) AS data_changed
FROM analysis_runs r
JOIN datasets d ON d.id = r.dataset_id
WHERE d.name = 'insurance.csv'
ORDER BY r.analyzed_at, r.id;

-- 3. Which kinds of problem are most common across all stored runs?
SELECT i.issue_type, i.severity, COUNT(*) AS occurrences, SUM(i.affected_rows) AS rows_affected
FROM issues i
JOIN analysis_runs r ON r.id = i.run_id
GROUP BY i.issue_type, i.severity
ORDER BY occurrences DESC;

-- 4. Datasets that are NOT ready for machine learning (grade C or worse in the latest run)
SELECT d.name, r.ml_score, r.ml_grade, r.analyzed_at
FROM analysis_runs r
JOIN datasets d ON d.id = r.dataset_id
WHERE r.id = (SELECT MAX(r2.id) FROM analysis_runs r2 WHERE r2.dataset_id = r.dataset_id)
  AND r.ml_grade IN ('C', 'D', 'F');

-- 5. The weakest quality dimension of each dataset's latest run
SELECT d.name,
       LEAST(r.completeness, r.uniqueness, r.validity, r.consistency) AS weakest_score,
       CASE LEAST(r.completeness, r.uniqueness, r.validity, r.consistency)
            WHEN r.validity     THEN 'validity'
            WHEN r.completeness THEN 'completeness'
            WHEN r.uniqueness   THEN 'uniqueness'
            ELSE 'consistency' END AS weakest_dimension
FROM analysis_runs r
JOIN datasets d ON d.id = r.dataset_id
WHERE r.id = (SELECT MAX(r2.id) FROM analysis_runs r2 WHERE r2.dataset_id = r.dataset_id);

-- 6. Columns that keep having problems run after run
SELECT d.name, i.column_name, i.issue_type, COUNT(DISTINCT i.run_id) AS runs_with_problem
FROM issues i
JOIN analysis_runs r ON r.id = i.run_id
JOIN datasets d ON d.id = r.dataset_id
WHERE i.column_name IS NOT NULL
GROUP BY d.name, i.column_name, i.issue_type
HAVING COUNT(DISTINCT i.run_id) > 1
ORDER BY runs_with_problem DESC;
