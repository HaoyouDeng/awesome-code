# https://github.com/Delgan/loguru

from loguru import logger

logger.debug("That's it, beautiful and simple logging!")

logger.add("file.log")

# logger.add("file_1.log", rotation="500 MB")    # Automatically rotate too big file
# logger.add("file_2.log", rotation="12:00")     # New file is created each day at noon
# logger.add("file_3.log", rotation="1 week")    # Once the file is too old, it's rotated

# logger.add("file_X.log", retention="10 days")  # Cleanup after some time

# logger.add("file_Y.log", compression="zip")    # Save some loved space

logger.debug("That's it, beautiful and simple logging!")

logger.info("If you're using Python {}, prefer {feature} of course!", 3.6, feature="f-strings")
logger.info("addsome ")
logger.success(f"train over.")
logger.info("awesome code")
