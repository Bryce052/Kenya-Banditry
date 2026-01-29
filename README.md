# Kenya Banditry Data Project

A  research project that collects, analyzes, and visualizes banditry incidents in Kenya using open-source intelligence (OSINT) methods, statistical analysis, and interactive dashboards.

## Project Overview

This project aims to track, analyze, and present data on banditry incidents in Kenya through a systematic approach that combines:
- **Data Collection**: Web scraping from open-source information sources
- **Data Analysis**: Statistical analysis using Python
- **Visualization**: Interactive dashboards built with Power BI
- **Dissemination**: Academic publications and a public-facing website

## Features

- **Automated Data Scraping**: Collects banditry incident data from multiple open-source platforms
- **Statistical Analysis**: Processes and analyzes patterns, trends, and correlations in banditry data
- **Interactive Dashboards**: Power BI visualizations for exploring temporal and spatial patterns
- **Research Publications**: Academic manuscripts and reports based on findings
- **Public Website**: Centralized platform hosting dashboards, reports, and project information

## Project Structure

```
kenya-banditry-project/
├── data/
│   ├── raw/              # Raw scraped data
│   ├── processed/        # Cleaned and processed datasets
│   └── exports/          # Data exports for Power BI
├── scrapers/
│   ├── news_scraper.py   # News website scrapers
│   ├── social_scraper.py # Social media data collection
│   └── utils.py          # Helper functions
├── analysis/
│   ├── data_cleaning.py  # Data preprocessing scripts
│   ├── analysis.py       # Statistical analysis
│   └── notebooks/        # Jupyter notebooks for exploration
├── dashboards/
│   ├── powerbi/          # Power BI dashboard files (.pbix)
│   └── exports/          # Dashboard images and PDFs
├── reports/
│   ├── manuscripts/      # Academic papers and drafts
│   └── briefs/           # Policy briefs and summaries
├── website/
│   ├── index.html        # Website files
│   ├── assets/           # CSS, JS, images
│   └── data/             # Data files for web display
└── docs/
    └── methodology.md    # Detailed methodology documentation
```

## Technologies Used

### Data Collection
- **Python** (Beautiful Soup, Scrapy, Selenium)
- **APIs** for social media and news aggregators
- **RSS feeds** for news monitoring

### Data Analysis
- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning and statistical modeling
- **matplotlib/seaborn** - Data visualization
- **geopandas** - Geospatial analysis

### Visualization & Reporting
- **Power BI** - Interactive dashboards
- **Jupyter Notebooks** - Analysis documentation
- **LaTeX/Markdown** - Report writing

### Website
- **HTML/CSS/JavaScript** - Frontend
- **Power BI Embedded** - Dashboard integration
- Static site hosting (GitHub Pages)

## Installation

### Prerequisites
- Python 3.8 or higher
- Power BI Desktop (for dashboard development)
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Bryce052/kenya-banditry-project.git
cd kenya-banditry-project
```

## Data Sources

The project collects data from multiple open-source platforms including:
- National and regional news websites
- Social media platforms (Twitter/X, Facebook)
- Government press releases and reports
- NGO incident reports
- Academic datasets

*Note: All data collection complies with terms of service and ethical research guidelines.*

## Key Metrics Tracked

- **Incident count**: Number of banditry events over time
- **Casualties**: Deaths and injuries reported
- **Location data**: Geographic distribution of incidents
- **Perpetrator information**: Where available
- **Response measures**: Security and government responses
- **Economic impact**: Livestock stolen, property damage

## Dashboards

The Power BI dashboards include:

1. **Overview Dashboard**: High-level trends and key statistics
2. **Geographic Analysis**: Heat maps and regional breakdowns
3. **Temporal Patterns**: Time-series analysis of incidents
4. **Impact Assessment**: Casualties, displacement, economic costs
5. **Social Network Analysis**: Involved actors, criminal groups, and linked events

## Publications

Research outputs from this project include:

- Academic journal articles on banditry patterns and drivers
- Policy briefs for government and development partners
- Technical reports on methodology and findings
- Conference presentations and posters

Publications are stored in the `reports/` directory.

## Website

The project website provides:
- Interactive dashboards embedded via Power BI
- Downloadable reports and data summaries
- Methodology documentation
- About the project and team
- Contact information and feedback mechanisms

## Contributing

We welcome contributions to improve data collection, analysis methods, and visualizations.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

Please ensure your code follows PEP 8 style guidelines and includes appropriate documentation.

## Ethics & Data Privacy

This project adheres to ethical research principles:
- Only publicly available information is collected
- Personal identifying information is anonymized
- Data is used solely for research and public awareness purposes
- Sensitive information is handled with appropriate care
- Analysis aims to inform policy, not stigmatize communities


## Contact

**Technical Lead**: Bryce Barthuly 
**Email**: BryceBarthuly@outlook.com
**Institution**: John Jay College of Criminal Justice

**Project Manager**: Jim Karani
**Email**: jimkarani@gmail.com
**Institution**: John Jay College of Criminal Justice


## Citation

If you use this data or methodology in your research, please cite:

```
Barthuly, B., & Karani, J., (2026). Kenya Banditry Data Project [Data set and analysis]. 
GitHub. https://github.com/Bryce052kenya-banditry-project
```

## Roadmap

### Current Phase
- ✅ Data collection pipeline established
- ✅ Analysis framework developed
- ✅ Power BI dashboards created
- ✅ Website wireframe

### Future Development
- [ ] Expanded geographic coverage
- [ ] API for data access
- [ ] Website launch

---

**Last Updated**: January 2026  
**Version**: 1.0.0
