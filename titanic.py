import pandas
import matplotlib.pyplot


def main():
    # init
    decade_map = {0: '0-9',
                  1: '10-19',
                  2: '20-29',
                  3: '30-39',
                  4: '40-49',
                  5: '50-59',
                  6: '60-69',
                  7: '70-79'}

    d = pandas.read_csv('data.csv')
    num_passengers = len(d)
    num_survivors = d.loc[:, 'Survived'].sum()
    d['Decade'] = d.loc[:, 'Age'].dropna().apply(lambda x: decade_map[x // 10])
    matplotlib.pyplot.style.use('ggplot')

    # 1.
    print("Titanic passengers:", num_passengers, '', sep='\n')

    # 2.
    print("Overall survivors:", str(num_survivors / num_passengers * 100) + " %\n", sep='\n')

    # 3.
    sex = d[['Survived', 'SexCode']].groupby('SexCode').sum()
    print("Male survivors:", str(sex.loc[0, 'Survived'] / num_passengers * 100) + " %", sep='\n')
    print("Female survivors:", str(sex.loc[1, 'Survived'] / num_passengers * 100) + " %\n", sep='\n')

    # 4.
    survivors_by_class = d[['PClass', 'Survived']].groupby('PClass').sum()
    passengers_by_class = d[['PClass', 'Survived']].groupby('PClass').count()

    print("Class survival statistics:")
    for c in survivors_by_class.index:
        s = survivors_by_class.loc[c, 'Survived']
        p = passengers_by_class.loc[c, 'Survived']
        print(s, 'out of', p, 'from', c if c != '*' else 'unknown', 'class survived -', s / p * 100, '%')

    # 5.
    survivors_by_decade = d[['Decade', 'Survived']].fillna('unknown').groupby('Decade').sum() / num_survivors * 100
    survivors_by_decade.plot(kind='bar', figsize=(10, 10)).set_ylabel('Percentage')
    matplotlib.pyplot.savefig('decades.png')

    # 6.


    # print(survived)
    # print(len(survived))
    # print(d.info())


if __name__ == "__main__":
    main()
