import logging


def log_confusion_matrix(cm, labels):
    """
    Logs a confusion matrix as ASCII using Python's logging module.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    # Header
    header = "Pred\\True | " + " | ".join(f"{lbl:^10}" for lbl in labels)
    separator = "-" * len(header)
    logger.info(separator)
    logger.info(header)
    logger.info(separator)

    # Rows
    for i, row in enumerate(cm):
        row_str = " | ".join(f"{val:^10}" for val in row)
        logger.info(f"{labels[i]:^10} | {row_str}")
    logger.info(separator)
