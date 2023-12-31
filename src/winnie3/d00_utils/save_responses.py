from sqlalchemy import MetaData, Table
from winnie3.d00_utils.setup_sql_connection import setup_sql_connection
import logging


def save_responses(case_responses):
    """ Writes the responses generated by Winnie to the MySQL database
    :param case_responses: A pandas dataframe that contains the candidate responses, ranks and case ID
    :return: None
    """

    conn = setup_sql_connection()

    # check to make sure that columns align
    metadata = MetaData(bind=None)
    table = Table('recommended_responses_3', metadata, autoload=True, autoload_with=conn)

    case_keys = [column.name for column in table.c]
    case_keys.remove('id')
    if set(case_keys) != set(list(case_responses.columns)):
        logging.error('[save_responses] ERROR: the columns for the case_responses do not match recommended_responses')
        logging.info('[recommended_responses table]' + str(case_keys))
        logging.info('[case_responses.columns]' + str(list(case_responses.columns)))
        return

    # add rows for the new responses
    case_responses.to_sql('recommended_responses_3', conn, if_exists='append', chunksize=1000, index=False)

    conn.dispose()
